"""
RAG (Retrieval-Augmented Generation) module for menu search.
Uses FAISS for vector similarity search over the menu database.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Global instances
_faiss_index = None
_embeddings_model = None


class MenuRAG:
    """
    RAG system for restaurant menu search and retrieval.
    Uses sentence embeddings and FAISS for efficient similarity search.
    """
    
    def __init__(self, menu_path: str = "menu.json"):
        self.menu_path = menu_path
        self.menu_data = None
        self.documents = []
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the RAG system with menu data."""
        try:
            # Load menu data
            self._load_menu()
            
            # Create document chunks
            self._create_documents()
            
            # Create embeddings and index
            self._create_index()
            
            self.is_initialized = True
            logger.info(f"MenuRAG initialized with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize MenuRAG: {e}")
            # Continue with fallback search
            self.is_initialized = False
    
    def _load_menu(self):
        """Load menu data from JSON file."""
        menu_file = Path(self.menu_path)
        if not menu_file.exists():
            menu_file = Path(__file__).parent.parent / self.menu_path
        
        with open(menu_file, "r") as f:
            self.menu_data = json.load(f)
        
        logger.info(f"Loaded menu from {menu_file}")
    
    def _create_documents(self):
        """Create searchable document chunks from menu data."""
        self.documents = []
        
        for category in self.menu_data.get("categories", []):
            category_name = category["name"]
            category_desc = category.get("description", "")
            
            for item in category.get("items", []):
                # Create a rich text document for each menu item
                doc_text = (
                    f"{item['name']} - {category_name}. "
                    f"{item.get('description', '')} "
                    f"Price: ${item['price']:.2f}. "
                )
                
                # Add dietary info
                dietary = item.get("dietary", [])
                if dietary:
                    doc_text += f"Dietary: {', '.join(dietary)}. "
                
                # Add popularity
                if item.get("popular"):
                    doc_text += "Popular item. "
                
                if item.get("chef_special"):
                    doc_text += "Chef's special. "
                
                # Add add-ons
                addons = item.get("addons", [])
                if addons:
                    addon_text = ", ".join([f"{a['name']} (+${a['price']})" for a in addons])
                    doc_text += f"Add-ons available: {addon_text}. "
                
                self.documents.append({
                    "text": doc_text,
                    "item": item,
                    "category": category_name,
                    "id": item["id"]
                })
        
        # Add special deals
        specials = self.menu_data.get("specials", {})
        for combo in specials.get("combo_deals", []):
            doc_text = f"Special Deal: {combo['name']}. {combo['description']} Price: ${combo['price']:.2f}"
            self.documents.append({
                "text": doc_text,
                "item": combo,
                "category": "Specials",
                "id": f"special-{combo['name'].lower().replace(' ', '-')}"
            })
        
        # Add daily specials
        daily = specials.get("daily_specials", {})
        for day, special in daily.items():
            doc_text = f"{day.capitalize()} Special: {special}"
            self.documents.append({
                "text": doc_text,
                "item": {"name": f"{day.capitalize()} Special", "description": special},
                "category": "Daily Specials",
                "id": f"daily-{day}"
            })
    
    def _create_index(self):
        """Create FAISS index from document embeddings."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            # Load embedding model
            model_name = os.getenv(
                "EMBEDDING_MODEL",
                "all-MiniLM-L6-v2"
            )
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            
            # Create embeddings
            texts = [doc["text"] for doc in self.documents]
            self.embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
            self.index.add(self.embeddings)
            
            logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
            
        except ImportError as e:
            logger.warning(f"FAISS or sentence-transformers not available: {e}")
            logger.info("Falling back to keyword search")
            self.index = None
    
    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search the menu for relevant items.
        
        Args:
            query: Search query (natural language)
            top_k: Number of results to return
        
        Returns:
            Formatted string with relevant menu items
        """
        if not self.is_initialized:
            self.initialize()
        
        if self.index is not None and self.embedding_model is not None:
            return self._vector_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)
    
    def _vector_search(self, query: str, top_k: int) -> str:
        """Perform vector similarity search."""
        try:
            import faiss
            
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            )
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    item = doc["item"]
                    results.append(
                        f"• {item['name']} ({doc['category']}) - ${item.get('price', 'N/A')}\n"
                        f"  {item.get('description', '')}"
                    )
            
            return "\n".join(results) if results else "No matching items found."
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> str:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc in self.documents:
            text_lower = doc["text"].lower()
            
            # Score based on word matches
            score = sum(1 for word in query_words if word in text_lower)
            
            # Boost for exact phrase match
            if query_lower in text_lower:
                score += 5
            
            # Boost for popular items
            if doc["item"].get("popular"):
                score += 0.5
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and take top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_docs = scored_docs[:top_k]
        
        # Format results
        results = []
        for _, doc in top_docs:
            item = doc["item"]
            results.append(
                f"• {item['name']} ({doc['category']}) - ${item.get('price', 'N/A')}\n"
                f"  {item.get('description', '')}"
            )
        
        return "\n".join(results) if results else "No matching items found."
    
    def get_item_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific menu item by name (fuzzy match)."""
        name_lower = name.lower()
        
        for doc in self.documents:
            item_name = doc["item"].get("name", "").lower()
            if name_lower in item_name or item_name in name_lower:
                return doc["item"]
        
        return None
    
    def get_category_items(self, category: str) -> List[Dict[str, Any]]:
        """Get all items in a category."""
        category_lower = category.lower()
        return [
            doc["item"] for doc in self.documents
            if doc.get("category", "").lower() == category_lower
        ]
    
    def get_full_menu(self) -> Dict[str, Any]:
        """Return the complete menu data."""
        return self.menu_data
    
    def get_popular_items(self) -> List[Dict[str, Any]]:
        """Get all popular items."""
        return [
            doc["item"] for doc in self.documents
            if doc["item"].get("popular")
        ]
    
    def get_dietary_options(self, restriction: str) -> List[Dict[str, Any]]:
        """Get items matching a dietary restriction."""
        restriction_lower = restriction.lower()
        return [
            doc["item"] for doc in self.documents
            if restriction_lower in [d.lower() for d in doc["item"].get("dietary", [])]
        ]


def create_menu_embeddings(menu_path: str = "menu.json") -> MenuRAG:
    """
    Factory function to create and initialize a MenuRAG instance.
    """
    rag = MenuRAG(menu_path)
    rag.initialize()
    return rag

