import re
from evaluation.runner import SYNONYM_MAP, normalize_text

SYNONYM_MAP["supported"] = {"supported", "support", "ships", "shipping", "available", "destinations", "offers", "eligible"}

from evaluation.runner import check_concept_present
text = "Aster & Row currently ships internationally only to Canada. Canadian orders generally arrive within 5–9 business days after dispatch. Import duties, taxes, and brokerage charges are not prepaid by Aster & Row."

print("Canada is supported with synonym:", check_concept_present("Canada is supported", text))
