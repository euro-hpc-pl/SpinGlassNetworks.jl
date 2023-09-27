using Documenter, SpinGlassNetworks

_pages = [
    "Introduction" => "index.md",
    "User guide" => "userguide.md",
    "API Reference" => "api.md"
]

# ============================
format = Documenter.HTML(edit_link = "master",
                         prettyurls = get(ENV, "CI", nothing) == "true",
)

# format = Documenter.LaTeX(platform="none")

makedocs(
    sitename="SpinGlassNetworks.jl",
    modules = [SpinGlassNetworks],
    pages = _pages,
    format = format
    )