using Documenter, SpinGlassNetworks
# makedocs(
#     modules=[SpinGlassTensors],
#     sitename="SpinGlassTensors.jl",
#     format=Documenter.LaTeX()
# )
makedocs(
    modules=[SpinGlassNetworks],
    sitename="SpinGlassNetworks.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    )
)
deploydocs(
    repo="github.com/euro-hpc-pl/SpinGlassNetworks.jl.git",
    devbranch="projectors-doc"
)
