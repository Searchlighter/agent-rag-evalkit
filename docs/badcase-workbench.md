# Badcase Workbench

The Badcase workbench stores category, severity, manual conclusion, root cause, owner,
responsible version, resolution, lifecycle status, and an activity history.

Supported lifecycle:

    open -> triaged -> fixed
                    -> ignored
    open -> ignored

A resolution is required before moving to fixed. Workbench queries can filter
by EvalRun, category, severity, status, or owner. The detail response includes
the current Badcase and all creation/status-change activities.

The in-memory repository is suitable for the Demo only. Production use needs
database persistence, pagination, authentication, and role-based permissions.
