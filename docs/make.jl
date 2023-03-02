using Documenter, SpinGlassNetworks

_pages = [
    "Home" => "index.md",
    "User guide" => "userguide.md",
    "API" => "api.md"
]

# ============================

makedocs(
    sitename="SpinGlassNetworks",
    modules = [SpinGlassNetworks],
    pages = _pages
    )