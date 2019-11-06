using Documenter, BridgeSPDE

makedocs(;
    modules=[BridgeSPDE],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/mschauer/BridgeSPDE.jl/blob/{commit}{path}#L{line}",
    sitename="BridgeSPDE.jl",
    authors="mschauer <moritzschauer@web.de>",
    assets=String[],
)

deploydocs(;
    repo="github.com/mschauer/BridgeSPDE.jl",
)
