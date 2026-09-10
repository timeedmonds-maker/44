"""Compatibility shim for legacy freeze_spin modules executed as a package.

Historical diagnostics imported this sibling by its old top-level module name.
Newer three-camera renderers import the diagnostics through the ``freeze_spin``
package, so expose the canonical implementation at the legacy name without
changing any reconstruction logic.
"""

from freeze_spin.build_portable_moge_pnp_freeview_v12 import *  # noqa: F401,F403
