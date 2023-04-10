x = int(evt.pos().x())
y = int(evt.pos().y())
self.last_click_x = x
self.last_click_y = y
self.run_code()
self.update_images()