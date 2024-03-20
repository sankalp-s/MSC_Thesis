import sys
import ctypes
import time
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key as keycodes
import retro

class SuperMarioInteractive:
    """
    Interactive setup for Super Mario World
    """
    def __init__(self, record):
        env = retro.make(game='SuperMarioWorld-Snes')
        self._buttons = env.buttons
        self._env = env
        self._actions_taken = []  # New attribute to store actions taken
        obs = env.reset()
        self._image = self.get_image(obs, env)

        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = self._image.shape[1]
        win_height = self._image.shape[0]

        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self._image.shape[1], self._image.shape[0], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self._win = win
        self._key_previous_states = {}

    def _update(self, dt):
        keys_clicked = set()
        keys_pressed = set()
        for key_code, pressed in self._key_handler.items():
            if pressed:
                keys_pressed.add(key_code)

            if not self._key_previous_states.get(key_code, False) and pressed:
                keys_clicked.add(key_code)
            self._key_previous_states[key_code] = pressed

        # Assume that for async environments, we just want to repeat keys for as long as they are held
        inputs = keys_pressed

        keys = []
        for keycode in inputs:
            for name in dir(keycodes):
                if getattr(keycodes, name) == keycode:
                    keys.append(name)

        act = self.keys_to_act(keys)

        obs, rew, done, _info = self._env.step(act)

        self._image = self.get_image(obs, self._env)
        
        # Append the action taken to the list
        self._actions_taken.append(act)

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(self._image.tobytes(), ctypes.POINTER(ctypes.c_short))
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self._image.shape[1], self._image.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, video_buffer)

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        self._env.close()
        sys.exit(0)

    def get_image(self, obs, env):
        return env.render(mode='rgb_array')

    def keys_to_act(self, keys):
        inputs = {
            None: False,

            'BUTTON': 'Z' in keys,
            'A': 'Z' in keys,
            'B': 'X' in keys,

            'C': 'C' in keys,
            'X': 'A' in keys,
            'Y': 'S' in keys,
            'Z': 'D' in keys,

            'L': 'Q' in keys,
            'R': 'W' in keys,

            'UP': 'UP' in keys,
            'DOWN': 'DOWN' in keys,
            'LEFT': 'LEFT' in keys,
            'RIGHT': 'RIGHT' in keys,

            'MODE': 'TAB' in keys,
            'SELECT': 'TAB' in keys,
            'RESET': 'ENTER' in keys,
            'START': 'ENTER' in keys,
        }
        return [inputs[b] for b in self._buttons]

class SuperMarioScreen:
    def __init__(self):
        self._interactive = SuperMarioInteractive(record=False)
        self._win = self._interactive._win

    def _update(self, dt):
        self._interactive._update(dt)

    def _draw(self):
        self._interactive._draw()

    def _on_close(self):
        self._interactive._on_close()

    def run(self):
        prev_frame_time = time.time()
        while True:
            self._win.switch_to()
            self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            self._draw()
            self._win.flip()

            # Print the state-action pair
            print("State:", self._interactive._image.shape)  # Modify this to print the actual state representation
            print("Action:", self._interactive._actions_taken[-1])  # Print the last action taken

if __name__ == '__main__':
    screen = SuperMarioScreen()
    screen.run()
