using Documenter, SpinGlassNetworks



makedocs(
    sitename="SpinGlassNetworks",
    modules = [SpinGlassNetworks],
    pages = [
        "Home" => "index.md",
        "User guide" => "userguide.md",
        "API" => "api.md"
    ]
    )

