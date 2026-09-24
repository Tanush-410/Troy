"""Static RBAC (C1/C3): a role's permission set P_r for the whole session.

Decisions depend only on the action name, never its parameters (see
policy/permissions.py on why parameter caps are out of scope).
"""

from __future__ import annotations

from policy.permissions import ROLE_PERMISSIONS, Role


def in_role(role: Role, action: str) -> bool:
    return action in ROLE_PERMISSIONS[role]


class StaticRBAC:
    def __init__(self, role: Role) -> None:
        self.role = role

    def allows(self, action: str) -> bool:
        return in_role(self.role, action)
