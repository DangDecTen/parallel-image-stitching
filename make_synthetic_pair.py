import numpy as np
from PIL import Image, ImageDraw, ImageFilter
rng = np.random.default_rng(1)
W,H = 2100,720
base = np.zeros((H,W,3),float)
for s,a in [(4,40),(16,60),(64,50)]:
    n = rng.random((H//s+2,W//s+2,3))
    im = Image.fromarray((n*255).astype(np.uint8)).resize((W,H),Image.BICUBIC)
    base += a*(np.asarray(im)/255.0-0.5)*2
base += 128
img = Image.fromarray(np.clip(base,0,255).astype(np.uint8))
d = ImageDraw.Draw(img)
for i in range(120):
    x,y = rng.integers(0,W),rng.integers(0,H); w,h = rng.integers(10,60,2)
    d.rectangle([x,y,x+w,y+h], fill=tuple(int(v) for v in rng.integers(0,255,3)))
img = img.filter(ImageFilter.GaussianBlur(0.8))
A = img.crop((0,0,1280,720))
c = (1.0, 0.03, 700.0,  -0.03, 1.0, 5.0,  2e-5, 0.0)    # B pixel -> base/A coords
B = img.transform((1280,720), Image.PERSPECTIVE, c, Image.BICUBIC)
def ppm(im,p): im.convert("RGB").save(p)
A.save("/tmp/A.ppm"); B.save("/tmp/B.ppm")
print("true H:", c)
