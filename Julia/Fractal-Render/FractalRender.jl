module FractalRender

  using CSFML.LibCSFML

  include("./state.jl")
  include("./colors.jl")
  include("./simulate.jl")
  include("./render.jl")

  function main()
    renderState = RenderState(width=512, height=512)
    simState = State(res=(512,512), iterations=100)

    event = Ref{sfEvent}()
    while Bool(sfRenderWindow_isOpen(renderState.window))
      while Bool(sfRenderWindow_pollEvent(renderState.window, event))
        if event.x.type == sfEvtClosed
          sfRenderWindow_close(renderState.window)
        end
      end

      step!(simState)

      render!(simState, renderState)
    end
  end
end

FractalRender.main()

