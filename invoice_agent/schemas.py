from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item: str
    normalized_item: str | None = None
    quantity: float
    unit_price: Decimal | None = None
    amount: Decimal | None = None
    note: str | None = None


class Invoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    invoice_number: str | None = None
    vendor: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    due_date_note: str | None = None
    currency: str = "USD"
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax_amount: Decimal | None = None
    shipping: Decimal | None = None
    total: Decimal | None = None
    payment_terms: str | None = None
    notes: str | None = None
    revision: str | None = None

    def to_state(self) -> dict:
        return self.model_dump(mode="json")


class Finding(BaseModel):
    severity: str
    code: str
    message: str
