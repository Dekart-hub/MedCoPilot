from __future__ import annotations

from unittest.mock import Mock

import di.container as container_module
from config import EhrSettings
from ehr import DisabledEhrGateway


def test_ehr_gateway_is_disabled_by_default():
    gateway = container_module._build_ehr_gateway(EhrSettings())

    assert isinstance(gateway, DisabledEhrGateway)


def test_enabled_ehr_builds_fhir_gateway_once(monkeypatch):
    settings = EhrSettings(enabled=True)
    gateway = object()
    factory = Mock(return_value=gateway)
    monkeypatch.setattr(container_module, "FhirR4EhrGateway", factory)

    result = container_module._build_ehr_gateway(settings)

    assert result is gateway
    factory.assert_called_once_with(settings)
