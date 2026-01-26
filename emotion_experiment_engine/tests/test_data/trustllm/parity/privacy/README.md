# TrustLLM Privacy Parity Fixtures

Synthetic fixtures with >=20 samples per branch mirroring TrustLLM privacy heuristics.
- `privacy_confAide_parity.json`: 50 responses spanning each mapping bucket + numeric detection.
- `privacy_awareness_query_parity.json`: 40 requests with expected longformer labels (`LABEL_0` refusal / `LABEL_1` comply).
- `privacy_leakage_parity.json`: 80 records covering all shot/type combinations with refusal/compliance annotations.
