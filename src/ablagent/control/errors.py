"""Typed control-plane failures."""

class ControlError(Exception):
    code = "control_error"

class InputError(ControlError):
    code = "invalid_input"

class IntegrityError(ControlError):
    code = "integrity_error"

class AuthorityError(ControlError):
    code = "authority_rejected"

class StateError(ControlError):
    code = "invalid_state"

class ConcurrentRevisionError(ControlError):
    code = "concurrent_revision"
