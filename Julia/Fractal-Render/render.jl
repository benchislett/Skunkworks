mutable struct RenderState
  mode::Any
  window::Any
  texture::Any
  sprite::Any
  background::Any
end

function RenderState(; width=32,height=32, background=sfColor_fromRGBA(0,0,0,255))
  mode = sfVideoMode(width, height, 32)

  window = sfRenderWindow_create(mode, "Julia Fractal Renderer", sfResize | sfClose, C_NULL)
  @assert window != C_NULL

  texture = sfTexture_create(width, height)
  @assert texture != C_NULL
  
  sprite = sfSprite_create()
  sfSprite_setTexture(sprite, texture, sfTrue)
  @assert sprite != C_NULL

  obj = RenderState(mode, window, texture, sprite, background)
  finalizer(destroy!, obj)
  return obj
end

function render!(simState::State, renderState::RenderState)
  sfRenderWindow_clear(renderState.window, renderState.background)
  sfTexture_updateFromPixels(renderState.texture, map(n -> getColor(simState, n), simState.fieldIterations), simState.res..., 0, 0)
  sfRenderWindow_drawSprite(renderState.window, renderState.sprite, C_NULL)
  sfRenderWindow_display(renderState.window)
end

function destroy!(renderState::RenderState)
  sfSprite_destroy(renderState.sprite)
  sfTexture_destroy(renderState.texture)
  sfRenderWindow_destroy(renderState.window)
end

