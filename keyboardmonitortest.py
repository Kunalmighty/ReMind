'''from pynput.keyboard import Key, Listener

def on_press(key):
    print("ya")
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        print('booty')
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

# Collect events until released
with Listener(on_press=on_press) as listener:
    listener.join()'''
from pynput import keyboard

def execute():
    print("YEAAA")

def on_press(key):
    if str(key) == 'v':
        execute()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()