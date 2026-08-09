"""Post-hoc causal-emergence downstream analysis.

Imports are intentionally lazy so lightweight audit/plot modules can be used without
initializing every model/data dependency.  The recommended comparison-complete entry
is ``scripts/run_unified_downstream_analysis.py``.
"""
__all__ = [
    "DownstreamConfig", "render_downstream_figures", "run_downstream_analysis",
    "UnifiedSuiteConfig", "build_unified_downstream_tables", "render_all_unified_figures",
]

def __getattr__(name):
    if name == "DownstreamConfig":
        from .config import DownstreamConfig
        return DownstreamConfig
    if name in {"render_downstream_figures", "run_downstream_analysis"}:
        from .workflow import render_downstream_figures, run_downstream_analysis
        return {"render_downstream_figures":render_downstream_figures,"run_downstream_analysis":run_downstream_analysis}[name]
    if name in {"UnifiedSuiteConfig", "build_unified_downstream_tables"}:
        from .unified_suite import UnifiedSuiteConfig, build_all_tables
        return {"UnifiedSuiteConfig":UnifiedSuiteConfig,"build_unified_downstream_tables":build_all_tables}[name]
    if name == "render_all_unified_figures":
        from .unified_plots import render_all_unified_figures
        return render_all_unified_figures
    raise AttributeError(name)
