"""
Session management for multi-turn conversations.
Supports in-memory storage and optional Redis for distributed deployments.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# Session timeout (30 minutes)
SESSION_TIMEOUT = 30 * 60


class SessionManager:
    """
    Manages conversation sessions for phone calls.
    Tracks order state, conversation history, and user context.
    """
    
    def __init__(self, use_redis: bool = False):
        """
        Initialize session manager.
        
        Args:
            use_redis: Whether to use Redis for session storage (for distributed deployments)
        """
        self.use_redis = use_redis and self._check_redis()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._redis_client = None
        
        if self.use_redis:
            self._init_redis()
        
        logger.info(f"SessionManager initialized (Redis: {self.use_redis})")
    
    def _check_redis(self) -> bool:
        """Check if Redis is available and configured."""
        try:
            import redis
            redis_url = os.getenv("REDIS_URL")
            return bool(redis_url)
        except ImportError:
            return False
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            import redis
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis_client = redis.from_url(redis_url)
            self._redis_client.ping()
            logger.info(f"Connected to Redis: {redis_url}")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.use_redis = False
    
    def get_or_create(self, session_id: str) -> Dict[str, Any]:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Unique session identifier (e.g., Twilio CallSid)
        
        Returns:
            Session data dictionary
        """
        session = self.get(session_id)
        
        if session is None:
            session = self._create_session(session_id)
            self.save(session_id, session)
        
        return session
    
    def _create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session with default values."""
        return {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "caller": None,
            "conversation_history": [],
            "current_order": [],
            "order_total": 0.0,
            "state": "greeting",  # greeting, browsing, ordering, confirming, complete
            "timeout_count": 0,
            "dietary_preferences": [],
            "metadata": {}
        }
    
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data or None if not found/expired
        """
        if self.use_redis:
            return self._get_redis(session_id)
        else:
            return self._get_memory(session_id)
    
    def _get_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from in-memory storage."""
        session = self._sessions.get(session_id)
        
        if session:
            # Check expiration
            last_activity = datetime.fromisoformat(session["last_activity"])
            if datetime.now() - last_activity > timedelta(seconds=SESSION_TIMEOUT):
                self.delete(session_id)
                return None
            
            # Update last activity
            session["last_activity"] = datetime.now().isoformat()
        
        return session
    
    def _get_redis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from Redis."""
        try:
            data = self._redis_client.get(f"session:{session_id}")
            if data:
                session = json.loads(data)
                session["last_activity"] = datetime.now().isoformat()
                self._redis_client.setex(
                    f"session:{session_id}",
                    SESSION_TIMEOUT,
                    json.dumps(session)
                )
                return session
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return self._get_memory(session_id)
    
    def save(self, session_id: str, session: Dict[str, Any]):
        """
        Save session data.
        
        Args:
            session_id: Session identifier
            session: Session data to save
        """
        session["last_activity"] = datetime.now().isoformat()
        
        if self.use_redis:
            self._save_redis(session_id, session)
        else:
            self._save_memory(session_id, session)
    
    def _save_memory(self, session_id: str, session: Dict[str, Any]):
        """Save session to in-memory storage."""
        self._sessions[session_id] = session
    
    def _save_redis(self, session_id: str, session: Dict[str, Any]):
        """Save session to Redis with expiration."""
        try:
            self._redis_client.setex(
                f"session:{session_id}",
                SESSION_TIMEOUT,
                json.dumps(session)
            )
        except Exception as e:
            logger.error(f"Redis save error: {e}")
            self._save_memory(session_id, session)
    
    def delete(self, session_id: str):
        """Delete a session."""
        if self.use_redis:
            try:
                self._redis_client.delete(f"session:{session_id}")
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        self._sessions.pop(session_id, None)
    
    def cleanup_expired(self):
        """Remove expired sessions (for in-memory storage)."""
        if self.use_redis:
            return  # Redis handles expiration automatically
        
        now = datetime.now()
        expired = []
        
        for session_id, session in self._sessions.items():
            last_activity = datetime.fromisoformat(session["last_activity"])
            if now - last_activity > timedelta(seconds=SESSION_TIMEOUT):
                expired.append(session_id)
        
        for session_id in expired:
            del self._sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def cleanup_all(self):
        """Clean up all sessions (for shutdown)."""
        if self.use_redis:
            try:
                keys = self._redis_client.keys("session:*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis cleanup error: {e}")
        
        self._sessions.clear()
        logger.info("All sessions cleaned up")
    
    def get_active_count(self) -> int:
        """Get count of active sessions."""
        if self.use_redis:
            try:
                return len(self._redis_client.keys("session:*"))
            except Exception:
                pass
        
        self.cleanup_expired()
        return len(self._sessions)
    
    def update_state(self, session_id: str, state: str):
        """Update session state."""
        session = self.get(session_id)
        if session:
            session["state"] = state
            self.save(session_id, session)
    
    def add_to_order(self, session_id: str, item: Dict[str, Any]):
        """Add an item to the session's order."""
        session = self.get(session_id)
        if session:
            session["current_order"].append(item)
            session["order_total"] += item.get("price", 0)
            self.save(session_id, session)
    
    def remove_from_order(self, session_id: str, item_name: str) -> bool:
        """Remove an item from the order by name."""
        session = self.get(session_id)
        if session:
            for i, item in enumerate(session["current_order"]):
                if item.get("name", "").lower() == item_name.lower():
                    removed = session["current_order"].pop(i)
                    session["order_total"] -= removed.get("price", 0)
                    self.save(session_id, session)
                    return True
        return False
    
    def add_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str
    ):
        """Add a turn to the conversation history."""
        session = self.get(session_id)
        if session:
            session["conversation_history"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            # Keep last 50 turns to prevent memory bloat
            session["conversation_history"] = session["conversation_history"][-50:]
            self.save(session_id, session)

