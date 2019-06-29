from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import uvicorn, aiohttp, asyncio
from io import BytesIO
import uuid
import time

from fastai import *
from fastai.text import *
from fastai.vision.gan import basic_generator
from fastai.vision.image import Image
import PIL


lm_file_url = 'https://www.dropbox.com/s/tov1ay47aricy3z/lm_export.pkl?dl=1'
lm_file_name = 'lm_export.pkl'
gan_file_url = 'https://www.dropbox.com/s/sdf8xze95se2ied/gan_gen.pkl?dl=1'
gan_file_name = 'gan_gen.pkl'

device = None
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory=path/'static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    #await download_file(lm_file_url, path/'models'/lm_file_name)
    #await download_file(gan_file_url, path/'models'/gan_file_name)
    try:
        lm_learn = load_learner(path/'models', lm_file_name)
        encoder = lm_learn.model[0].module
        encoder.reset()
        _ = encoder.eval()
        global device
        device = one_param(encoder).device
        
        gan_gen = Txt2ImgGenerator()
        gan_gen.load_state_dict(torch.load(path/'models'/f'{gan_file_name}'))
        _ = gan_gen.eval()
        gan_gen.to(device)
        return encoder, lm_learn.data, gan_gen
    except RuntimeError as e:
        raise

def encode(description):
    xb, _ = lm_data.one_item(description)
    return encoder(xb)[0][-1][:,-1].view(400,1,1)


# TODO: can put "squeezer" and "Txt2ImgGenerator" into a file and import
def squeezer(in_dim, out_dim):
    return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

class Txt2ImgGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = basic_generator(in_size=64, n_channels=3, noise_sz=228)
        self.squeezer = squeezer(400, 128)
    
    def forward(self, embedding, fake_image=None):        
        em_s = self.squeezer(embedding.view(embedding.size(0), -1))
        em_s = em_s[:,:,None,None]
        em_noise = torch.cat([em_s, torch.randn(em_s.size(0),100,1,1).cuda()], 1)
        # return: (embedding, fake image)
        return embedding, self.generator(em_noise)

def gen_image(embedding):
    with torch.no_grad():
        _, img = gan_gen(embedding[None])
        return Image(img[0].float().clamp(min=0,max=1))

# For a different arch, need to use its own generator class
# class Txt2ImgGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.generator = basic_generator(in_size=64, n_channels=3, noise_sz=500)
#   
#     def forward(self, em_noise, fake_image=None):
#         # return: (embedding, fake image)
#         return em_noise[:,:400,:,:], self.generator(em_noise)

# def gen_image(embedding):
#     with torch.no_grad():
#         x = torch.cat([embedding, torch.randn(100,1,1).to(device)], 0)[None]
#         _, img = gan_gen(x)
#         return Image(img[0].float().clamp(min=0,max=1))

# Send images in memory. Not convenient to send multiple images.
# def serve_pil_image(pil_img):
#     img_io = BytesIO()
#     pil_img.save(img_io, 'PNG')
#     img_io.seek(0)
#     return send_bytes(img_io, mimetype='image/png')

async def cleanup(fns):
    await asyncio.sleep(1)
    for fn in fns:
        (path/fn[3:]).unlink()


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
encoder, lm_data, gan_gen = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/gen_img', methods=['POST'])
async def gen_img(request):    
    data = await request.form()
    description = data['description']
    embedding = encode(description)
    fns = []
    for _ in range(9):
        rdn = str(uuid.uuid4()).split('-')[0]
        fn = f'static/images/tmp_{rdn}.jpg'
        gen_image(embedding).save(path/fn)
        fns.append('../' + fn)
    task = BackgroundTask(cleanup, fns=fns)
    return JSONResponse({'url': fns}, background=task)


if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
