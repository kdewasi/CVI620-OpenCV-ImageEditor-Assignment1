import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# --------------------
# Part I: Draw OpenCV logo
# --------------------
def draw_opencv_logo(size=512, output_path='opencv_logo.png'):
    # create white canvas
    image = np.full((size, size, 3), 255, dtype=np.uint8)

    # put "OpenCV" text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 * (size / 512)
    font_th = int(6 * (size / 512))
    text = 'OpenCV'
    # estimate text size and position
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_th)
    text_x = (size - tw) // 2
    text_y = int(size * 0.14)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,0,0), font_th, cv2.LINE_AA)

    # parameters for ellipses
    outer_axes = (int(size * 0.175), int(size * 0.175))  # radius for ring
    inner_axes = (int(size * 0.058), int(size * 0.058))  # carve-out radius
    sweep = 300  # degrees swept by each colored arc
    center_positions = [
        (size//2, int(size * 0.4)),       # top-center
        (int(size * 0.31), int(size * 0.75)), # bottom-left
        (int(size * 0.69), int(size * 0.75))  # bottom-right
    ]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue
    rotation_offsets = [120, 0, 300]

    # draw and carve each arc
    for (cx, cy), clr, rot in zip(center_positions, colors, rotation_offsets):
        # draw filled outer ellipse
        cv2.ellipse(image, (cx, cy), outer_axes, rot, 0, sweep, clr, -1, cv2.LINE_AA)
        # carve out inner ellipse (white)
        cv2.ellipse(image, (cx, cy), inner_axes, rot, 0, sweep, (255,255,255), -1, cv2.LINE_AA)

    # save output
    cv2.imwrite(output_path, image)
    print(f"Saved OpenCV logo to {output_path}")

# --------------------
# Part II: Manual blend
# --------------------
def manual_blend():
    path1 = input("Path to first image: ")
    path2 = input("Path to second image: ")
    alpha = float(input("Alpha (0.0–1.0): "))

    # load images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # resize img2 to img1’s size
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))

    # convert to floats [0,1]
    img1_f = img1.astype(np.float32) / 255.0
    img2_f = img2.astype(np.float32) / 255.0

    # blend
    blended = (1 - alpha) * img1_f + alpha * img2_f
    out = (blended * 255).astype(np.uint8)

    # save & show
    cv2.imwrite('manual_blend.jpg', out)
    print("Saved manual_blend.jpg")
    cv2.imshow('Blended', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------
# Part III: Mini photo editor
# --------------------
class PhotoEditor:
    def __init__(self, img_path):
        self.stack = deque()
        self.log = []
        img = cv2.imread(img_path)
        self.current = img
        self.stack.append(img.copy())
        print("Loaded image:", img_path)

    def show_side_by_side(self):
        before = self.stack[-1]
        after = self.current
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1); plt.title('Before'); plt.axis('off')
        plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        plt.subplot(1,2,2); plt.title('After'); plt.axis('off')
        plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
        plt.show()

    def adjust_brightness(self):
        beta = int(input("Brightness offset (−100 to 100): "))
        self.stack.append(self.current.copy())
        self.current = cv2.convertScaleAbs(self.current, alpha=1.0, beta=beta)
        self.log.append(f"brightness {beta}")
        self.show_side_by_side()

    def adjust_contrast(self):
        alpha = float(input("Contrast factor (0.5 to 3.0): "))
        self.stack.append(self.current.copy())
        self.current = cv2.convertScaleAbs(self.current, alpha=alpha, beta=0)
        self.log.append(f"contrast {alpha}")
        self.show_side_by_side()

    def convert_grayscale(self):
        self.stack.append(self.current.copy())
        gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        self.current = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.log.append("grayscale")
        self.show_side_by_side()

    def add_padding(self):
        pad = int(input("Padding size (px): "))
        btype = input("Border type [constant, reflect, replicate, wrap]: ").lower()
        border_map = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE,
            'wrap': cv2.BORDER_WRAP
        }
        b = border_map.get(btype, cv2.BORDER_CONSTANT)
        self.stack.append(self.current.copy())
        self.current = cv2.copyMakeBorder(self.current, pad, pad, pad, pad, b, value=[0,0,0])
        self.log.append(f"padded {pad}px {btype}")
        self.show_side_by_side()

    def apply_threshold(self):
        mode = input("binary or inverse? ").lower()
        ttype = cv2.THRESH_BINARY if mode=='binary' else cv2.THRESH_BINARY_INV
        _, thr = cv2.threshold(cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY),127,255, ttype)
        self.stack.append(self.current.copy())
        self.current = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        self.log.append(f"threshold {mode}")
        self.show_side_by_side()

    def blend_with_image(self):
        path = input("Second image path: ")
        alpha = float(input("Alpha (0–1): "))
        other = cv2.imread(path)
        h, w = self.current.shape[:2]
        other = cv2.resize(other, (w,h)).astype(np.float32)/255.0
        base  = self.current.astype(np.float32)/255.0
        res   = cv2.addWeighted(base,1-alpha,other,alpha,0)
        self.stack.append(self.current.copy())
        self.current = (res*255).astype(np.uint8)
        self.log.append(f"blend {path} {alpha}")
        self.show_side_by_side()

    def undo(self):
        if len(self.stack)>1:
            self.current = self.stack.pop()
            print("Undid:", self.log.pop())
        else:
            print("Nothing to undo.")

    def view_history(self):
        print("History of operations:")
        for i,e in enumerate(self.log,1): print(f"  {i}. {e}")

    def save_and_exit(self):
        fname = input("Filename to save (e.g., result.jpg): ")
        cv2.imwrite(fname, self.current)
        print("Saved as", fname)
        self.view_history()
        exit(0)

    def run(self):
        menu = {
            '1': self.adjust_brightness,
            '2': self.adjust_contrast,
            '3': self.convert_grayscale,
            '4': self.add_padding,
            '5': self.apply_threshold,
            '6': self.blend_with_image,
            '7': self.undo,
            '8': self.view_history,
            '9': self.save_and_exit
        }
        while True:
            print("""
==== Mini Photo Editor ====
1. Adjust Brightness
2. Adjust Contrast
3. Convert to Grayscale
4. Add Padding
5. Apply Thresholding
6. Blend with Another Image
7. Undo Last Operation
8. View History of Operations
9. Save and Exit
""")
            choice=input("Select an option: ")
            if choice in menu: menu[choice]()
            else: print("Invalid choice.")

# --------------------
# Main entry point
# --------------------
def main():
    print("""
Choose:
1. Draw OpenCV logo
2. Manual blend two images
3. Run mini photo editor
""")
    c=input("Option: ")
    if c=='1': draw_opencv_logo()
    elif c=='2': manual_blend()
    elif c=='3': PhotoEditor(input("Path to image to edit: ")).run()
    else: print("Goodbye.")

if __name__=='__main__':
    main()
