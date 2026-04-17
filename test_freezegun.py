import freezegun
import importlib.metadata
with freezegun.freeze_time("2020-01-01"):
    print(importlib.metadata.version('email-validator'))
