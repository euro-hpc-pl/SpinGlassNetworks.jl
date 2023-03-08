using Documenter, SpinGlassNetworks

_pages = [
    "Introduction" => "index.md",
    "User guide" => "userguide.md",
    "API Reference" => "api.md"
]

# ============================

makedocs(
    sitename="SpinGlassNetworks",
    modules = [SpinGlassNetworks],
    pages = _pages
    )