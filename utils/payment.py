"""
Payment processing utilities.
Simulates POS/payment integration with Stripe-like functionality.
"""

import os
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PaymentStatus(Enum):
    """Payment status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class PaymentProcessor:
    """
    Simulated payment processor.
    In production, this would integrate with Stripe, Square, or other payment providers.
    """
    
    def __init__(self):
        """Initialize payment processor."""
        self.stripe_api_key = os.getenv("STRIPE_API_KEY")
        self.use_stripe = bool(self.stripe_api_key)
        
        # In-memory payment records (for simulation)
        self._payments: Dict[str, Dict[str, Any]] = {}
        
        if self.use_stripe:
            logger.info("Payment processor initialized with Stripe")
        else:
            logger.info("Payment processor initialized in simulation mode")
    
    async def process_payment(
        self,
        order_id: str,
        amount: float,
        payment_method: str = "card",
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a payment for an order.
        
        Args:
            order_id: Associated order ID
            amount: Payment amount in dollars
            payment_method: Payment method (card, cash, etc.)
            customer_id: Optional customer identifier
        
        Returns:
            Payment result dictionary
        """
        payment_id = f"pay_{uuid.uuid4().hex[:16]}"
        
        logger.info(f"Processing payment {payment_id} for order {order_id}: ${amount:.2f}")
        
        if self.use_stripe and payment_method == "card":
            return await self._process_stripe_payment(
                payment_id, order_id, amount, customer_id
            )
        else:
            return self._simulate_payment(payment_id, order_id, amount, payment_method)
    
    async def _process_stripe_payment(
        self,
        payment_id: str,
        order_id: str,
        amount: float,
        customer_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Process payment through Stripe.
        
        Note: In a real implementation, this would:
        1. Create a PaymentIntent
        2. Handle 3D Secure if required
        3. Capture the payment
        """
        try:
            import stripe
            stripe.api_key = self.stripe_api_key
            
            # Create payment intent
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Stripe uses cents
                currency="usd",
                metadata={
                    "order_id": order_id,
                    "payment_id": payment_id
                },
                customer=customer_id,
                automatic_payment_methods={"enabled": True}
            )
            
            payment_record = {
                "id": payment_id,
                "stripe_id": intent.id,
                "order_id": order_id,
                "amount": amount,
                "status": PaymentStatus.PROCESSING.value,
                "method": "card",
                "created_at": datetime.now().isoformat()
            }
            
            self._payments[payment_id] = payment_record
            
            logger.info(f"Stripe PaymentIntent created: {intent.id}")
            
            return {
                "success": True,
                "payment_id": payment_id,
                "stripe_client_secret": intent.client_secret,
                "status": PaymentStatus.PROCESSING.value
            }
            
        except ImportError:
            logger.warning("Stripe library not installed, using simulation")
            return self._simulate_payment(payment_id, order_id, amount, "card")
        except Exception as e:
            logger.error(f"Stripe payment failed: {e}")
            return {
                "success": False,
                "payment_id": payment_id,
                "error": str(e),
                "status": PaymentStatus.FAILED.value
            }
    
    def _simulate_payment(
        self,
        payment_id: str,
        order_id: str,
        amount: float,
        payment_method: str
    ) -> Dict[str, Any]:
        """
        Simulate a payment (for demo/testing).
        """
        # Simulate processing time and success
        import random
        import time
        
        # Simulate slight delay
        time.sleep(0.5)
        
        # 95% success rate for simulation
        success = random.random() < 0.95
        
        payment_record = {
            "id": payment_id,
            "order_id": order_id,
            "amount": amount,
            "method": payment_method,
            "status": PaymentStatus.COMPLETED.value if success else PaymentStatus.FAILED.value,
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat() if success else None,
            "simulated": True
        }
        
        self._payments[payment_id] = payment_record
        
        if success:
            logger.info(f"Payment {payment_id} completed successfully (simulated)")
        else:
            logger.warning(f"Payment {payment_id} failed (simulated)")
        
        return {
            "success": success,
            "payment_id": payment_id,
            "status": payment_record["status"],
            "message": "Payment processed" if success else "Payment declined"
        }
    
    def get_payment(self, payment_id: str) -> Optional[Dict[str, Any]]:
        """Get payment details by ID."""
        return self._payments.get(payment_id)
    
    def get_payments_for_order(self, order_id: str) -> List[Dict[str, Any]]:
        """Get all payments for an order."""
        return [
            p for p in self._payments.values()
            if p.get("order_id") == order_id
        ]
    
    async def refund_payment(
        self,
        payment_id: str,
        amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Refund a payment (full or partial).
        
        Args:
            payment_id: Payment to refund
            amount: Refund amount (None for full refund)
        
        Returns:
            Refund result
        """
        payment = self._payments.get(payment_id)
        
        if not payment:
            return {"success": False, "error": "Payment not found"}
        
        if payment["status"] != PaymentStatus.COMPLETED.value:
            return {"success": False, "error": "Payment not in completed status"}
        
        refund_amount = amount or payment["amount"]
        
        if self.use_stripe and payment.get("stripe_id"):
            return await self._refund_stripe(payment, refund_amount)
        else:
            return self._simulate_refund(payment_id, refund_amount)
    
    async def _refund_stripe(
        self,
        payment: Dict[str, Any],
        amount: float
    ) -> Dict[str, Any]:
        """Process Stripe refund."""
        try:
            import stripe
            stripe.api_key = self.stripe_api_key
            
            refund = stripe.Refund.create(
                payment_intent=payment["stripe_id"],
                amount=int(amount * 100)
            )
            
            payment["status"] = PaymentStatus.REFUNDED.value
            payment["refund_id"] = refund.id
            payment["refunded_at"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "refund_id": refund.id,
                "amount": amount
            }
            
        except Exception as e:
            logger.error(f"Stripe refund failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _simulate_refund(
        self,
        payment_id: str,
        amount: float
    ) -> Dict[str, Any]:
        """Simulate a refund."""
        payment = self._payments.get(payment_id)
        
        if payment:
            payment["status"] = PaymentStatus.REFUNDED.value
            payment["refunded_at"] = datetime.now().isoformat()
            payment["refund_amount"] = amount
        
        logger.info(f"Refund processed for payment {payment_id}: ${amount:.2f} (simulated)")
        
        return {
            "success": True,
            "refund_id": f"ref_{uuid.uuid4().hex[:12]}",
            "amount": amount,
            "simulated": True
        }


class POSIntegration:
    """
    Simulated POS (Point of Sale) integration.
    Handles order routing to kitchen displays, receipts, etc.
    """
    
    def __init__(self):
        """Initialize POS integration."""
        self.pos_endpoint = os.getenv("POS_ENDPOINT")
        self._order_queue = []
    
    async def send_to_kitchen(self, order: Dict[str, Any]) -> bool:
        """
        Send order to kitchen display system.
        
        In production, this would integrate with:
        - Toast POS
        - Square for Restaurants
        - Clover
        - Custom kitchen display systems
        """
        try:
            # In production, send HTTP request to POS API
            if self.pos_endpoint:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.pos_endpoint}/orders",
                        json=order
                    ) as response:
                        return response.status == 200
            
            # Simulation: add to queue
            self._order_queue.append({
                "order": order,
                "received_at": datetime.now().isoformat(),
                "status": "received"
            })
            
            logger.info(f"Order {order.get('id')} sent to kitchen (simulated)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send order to kitchen: {e}")
            return False
    
    async def print_receipt(
        self,
        order: Dict[str, Any],
        printer_id: str = "default"
    ) -> bool:
        """
        Print receipt for an order.
        
        In production, this would send to a receipt printer via:
        - Star Micronics CloudPRNT
        - Epson ePOS
        - Direct IP printing
        """
        receipt_text = self._generate_receipt(order)
        
        logger.info(f"Receipt printed for order {order.get('id')} (simulated)")
        logger.debug(f"Receipt:\n{receipt_text}")
        
        return True
    
    def _generate_receipt(self, order: Dict[str, Any]) -> str:
        """Generate receipt text."""
        lines = [
            "=" * 40,
            "BELLA'S ITALIAN KITCHEN".center(40),
            "=" * 40,
            f"Order: {order.get('id', 'N/A')}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "-" * 40,
        ]
        
        for item in order.get("items", []):
            name = item.get("name", "Unknown")
            price = item.get("price", 0)
            lines.append(f"{name:<30} ${price:.2f}")
            
            for custom in item.get("customizations", []):
                lines.append(f"  + {custom}")
        
        lines.extend([
            "-" * 40,
            f"{'Subtotal:':<30} ${order.get('total', 0):.2f}",
            f"{'Tax (8.5%):':<30} ${order.get('total', 0) * 0.085:.2f}",
            f"{'TOTAL:':<30} ${order.get('total', 0) * 1.085:.2f}",
            "=" * 40,
            "Thank you for your order!".center(40),
            "=" * 40,
        ])
        
        return "\n".join(lines)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get kitchen queue status."""
        return {
            "pending": len([o for o in self._order_queue if o["status"] == "received"]),
            "in_progress": len([o for o in self._order_queue if o["status"] == "preparing"]),
            "completed": len([o for o in self._order_queue if o["status"] == "ready"])
        }


# Type hint fix
from typing import List

