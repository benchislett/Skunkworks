# Efficient Scientific Computing in Julia (OIST) 10am-5:30pm Wednesday July 17, 2019
# by Valentin Churavy, JuliaLab, CSAIL, MIT

# Documentation: https://docs.julialang.org/en/stable
# Forum: https://discourse.julialang.org
# ThinkJulia: https://benlauwens.github.io/ThinkJulia.jl/latest/book.html
# Packages: https://juliaobserver.org and https://pkg.julialang.org
# Videos: https://youtube.com/TheJuliaLanguage

#=
Best practices for Scientific Code

What is reproducible science?
Questions to ask yourself:
- Can your code be run on a different machine?
- What about a different OS?

Scientific code needs:
- Tests
- Documentation
- Version control

Separate library code from "analysis" code.
Use notebooks for analysis.

Version control with Github
- Put your code online!
- Add a license (MIT/Apache/GPL)
- Use PRs, even on solo projects
- Look into Zenodo
=#

#=
Packages in Julia
https://github.com/JuliaLang/Example.jl

Example.jl
docs/make.jl
docs/src/index.md
src/Example.jl
test/runtests.jl

# src/Example.jl
module Example
  export hello_world

  """
  hello_world()
  
  Returns the standard Hello world string
  """
  function hello_world()
    return "Hello world!"
  end

end

# test/runtests.jl
using Test, Example
@test hello_world() == "Hello World!"

# docs/make.jl
using Documenter, Example

makedocs(modules = [Example], sitename = "Example.jl")

deploydocs(
    repo = "github.com/JuliaLang/Example.jl.git",
)
=#

#=
Package manager
"]" key in interactive terminal
- st (Status)
- generate ProjectName (Make a new project and create Package.toml)
- activate dir (Start using project at dir, used to switch environments)
- add PackageName (Install package and add to Package.toml)
- resolve (Install from Package.toml and create Manifest)
- instantiate (install from Manifest)
- ? (For details on all commands)

Package.toml: Package info with dependency info
Manifest.toml: Lockfile for dependency reproducability
Versions.toml: Version list for dependencies

julia --project=.
Run julia with a given project folder:
- Any installed packages are saved into Package.toml and Example.toml

=#


