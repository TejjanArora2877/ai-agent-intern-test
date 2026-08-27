"""Order lookup tool with data-layer privacy enforcement and status precedence rules."""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Set

from src.config import ORDERS_FILE
from src.models.schemas import CustomerSafeOrderView, OrderItemView

# Invariant: Prohibited sensitive fields that must NEVER enter LLM context
PROHIBITED_FIELDS: Set[str] = {"customer", "internal"}

# Regex pattern for matching order IDs (e.g. ORD-1001)
ORDER_ID_PATTERN = re.compile(r"\b(ORD-\d+)\b", re.IGNORECASE)


def extract_order_id(text: str) -> Optional[str]:
    """Extract an order ID from unstructured text, normalizing harmless differences."""
    if not text:
        return None
    match = ORDER_ID_PATTERN.search(text)
    if match:
        return match.group(1).upper().strip()
    return None


def normalize_order_id(order_id_input: str) -> Optional[str]:
    """Normalize input order ID (strip whitespace, punctuation, uppercase)."""
    if not order_id_input:
        return None
    # Strip harmless surrounding punctuation and whitespace
    cleaned = order_id_input.strip(" \t\n\r.,!?:;\"'()[]{}")
    match = ORDER_ID_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()
    return cleaned.upper() if cleaned else None


class OrderLookupTool:
    """Tool for querying orders from orders.json with strict privacy and status invariants."""

    def __init__(self, orders_file: Optional[Path] = None):
        self.orders_file = orders_file or ORDERS_FILE
        self._orders_by_id: Dict[str, Dict[str, Any]] = {}
        self._snapshot_at: Optional[str] = None
        self._load_orders()

    def _load_orders(self) -> None:
        """Load and index orders from JSON file."""
        if not self.orders_file.exists():
            return

        with open(self.orders_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._snapshot_at = data.get("snapshot_at")
        for order in data.get("orders", []):
            oid = order.get("order_id")
            if oid:
                self._orders_by_id[oid.upper()] = order

    @property
    def snapshot_at(self) -> Optional[str]:
        """Snapshot time for order calculations."""
        return self._snapshot_at

    def lookup(self, order_id_input: str) -> CustomerSafeOrderView:
        """
        Perform a safe lookup of an order by ID.
        
        Enforces:
        1. Input normalization (lowercase/whitespace/punctuation handling).
        2. Strict PII / internal data stripping at the Python data boundary.
        3. Authoritative status precedence (suppressing stale ETA/carrier on cancelled/returned).
        4. Safe handling of unknown and exception states.
        """
        normalized_id = normalize_order_id(order_id_input)
        if not normalized_id:
            return CustomerSafeOrderView(
                order_id=order_id_input or "",
                status="unknown",
                message="Please provide a valid order ID (e.g., ORD-1001).",
                handoff_recommended=False,
            )

        raw_order = self._orders_by_id.get(normalized_id)
        if not raw_order:
            return CustomerSafeOrderView(
                order_id=normalized_id,
                status="not_found",
                message=f"Order {normalized_id} was not found in our system. Please double-check your order ID or contact customer support.",
                handoff_recommended=True,
            )

        # Build customer-safe items list
        safe_items = []
        for item in raw_order.get("items", []):
            safe_items.append(
                OrderItemView(
                    sku=item.get("sku"),
                    name=item.get("name", "Unknown Item"),
                    quantity=item.get("quantity", 1),
                    final_sale=bool(item.get("final_sale", False)),
                )
            )

        status = raw_order.get("status", "unknown")
        carrier = raw_order.get("carrier")
        tracking_number = raw_order.get("tracking_number")
        estimated_delivery = raw_order.get("estimated_delivery")
        customer_safe_message = raw_order.get("customer_safe_message")

        # STATUS PRECEDENCE CONTRACT RULES:
        # Rule 1: Stale carrier/tracking/ETA fields must be suppressed when cancelled or returned
        if status in ("cancelled", "returned"):
            carrier = None
            tracking_number = None
            estimated_delivery = None

        # Rule 2: If shipped but estimated_delivery is null, mark estimate as unavailable
        estimated_delivery_available = True
        if status == "shipped" and estimated_delivery is None:
            estimated_delivery_available = False

        # Rule 3: If status is exception, mandate support review and handoff
        requires_support_review = (status == "exception")
        handoff_recommended = (status == "exception")

        return CustomerSafeOrderView(
            order_id=raw_order.get("order_id", normalized_id),
            membership_tier=raw_order.get("membership_tier"),
            items=safe_items,
            placed_at=raw_order.get("placed_at"),
            status=status,
            status_updated_at=raw_order.get("status_updated_at"),
            shipped_at=raw_order.get("shipped_at"),
            delivered_at=raw_order.get("delivered_at"),
            carrier=carrier,
            tracking_number=tracking_number,
            estimated_delivery=estimated_delivery,
            estimated_delivery_available=estimated_delivery_available,
            customer_safe_message=customer_safe_message,
            requires_support_review=requires_support_review,
            handoff_recommended=handoff_recommended,
        )
