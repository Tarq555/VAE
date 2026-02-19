import os
import sys
import shutil
import math
import argparse
import random
import time
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
from imageio import imwrite
import torch
from torch import optim, nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models import make_model, DualBranchDiscriminator
from criteria.lpips import lpips
import piq
from pytorch_msssim import ssim, ms_ssim
from gaussian_diffusion import create_gaussian_diffusion 
from communication_sys import Channel, RIS, Beamforming, SWIPT, DiffusionModel, RSMA, DynamicPowerControl, TimeAwareCommunication, SemanticAccuracy, PrivacyProtection
from VAE import Generator, Discriminator, VAE  # استيراد النماذج
import torch
import numpy as np
from torch import nn

data_directory = '/content/GAN_SeCom/data'
class SimpleModel(nn.Module):
  def __init__(self):
      super(SimpleModel, self).__init__()
      self.fc = nn.Linear(3 * 32 * 32, 10)  # طبقة خطية (توقع 10 فئات)

  def forward(self, x):
      return self.fc(x.view(x.size(0), -1))  # تسطيح الصورة
# data_directory = '/content/GAN_SeCom/data'
def load_data(data_directory, batch_size):
  # تحويلات لتحسين جودة الصور
  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
  ])

  # تحميل البيانات مع تجاهل المجلدات غير المرغوب فيها
  dataset = datasets.ImageFolder(root=data_directory, transform=transform)
  
  # إنشاء DataLoader
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return data_loader
def train_and_save_model(data_dir, model_file_path, num_epochs=2, batch_size=32):
  model = SimpleModel()
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  criterion = nn.CrossEntropyLoss()  # استخدام خسارة التصنيف

  batch_size = 32  # يمكنك تغيير حجم الدفعة حسب الحاجة
  data_loader = load_data(data_directory, batch_size)
  for epoch in range(num_epochs):
      model.train()  # وضع النموذج في وضع التدريب
      running_loss = 0.0
      
      for images, labels in data_loader:
          optimizer.zero_grad()  # إعادة تعيين التدرجات
          outputs = model(images)  # تمرير البيانات عبر النموذج
          loss = criterion(outputs, labels)  # حساب الخسارة
          loss.backward()  # حساب التدرجات
          optimizer.step()  # تحديث الأوزان
          
          running_loss += loss.item()

      print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(data_loader)}")

  # حفظ النموذج
  torch.save(model.state_dict(), model_file_path)
  print(f"Model saved to {model_file_path}")

# تحويل التنسور إلى صورة
def tensor2image(tensor):
  images = tensor.cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()
  images = images * 127.5 + 127.5
  images = images.astype(np.uint8)
  return images

# إعداد السجل
def get_logger(filename, verbosity=1, name=None):
  level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
  formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
  logger = logging.getLogger(name)
  logger.setLevel(level_dict[verbosity])
  fh = logging.FileHandler(filename, "w")
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  return logger
class PowerNormalize(nn.Module):
    def __init__(self, t_pow=1):
        super(PowerNormalize, self).__init__()
        self.t_pow = t_pow

    def forward(self, x, dim=(1, 2)):
        pwr = torch.mean(x ** 2, dim, True)
        return np.sqrt(self.t_pow) * x / torch.sqrt(pwr)
def adaptive_coding_modulation(snr_db):
    # Define thresholds for SNR and corresponding coding & modulation schemes
    snr_db=60
    if snr_db > 20:
        modulation = '64-QAM'  # Higher throughput for high SNR
        coding_rate = 0.9
    elif snr_db > 10:
        modulation = '16-QAM'
        coding_rate = 0.7
    else:
        modulation = 'QPSK'  # More robust for low SNR
        coding_rate = 0.5

    print(f"Using modulation: {modulation} with coding rate: {coding_rate}")
    return modulation, coding_rate

def modulate(input_data, modulation_type):
    if modulation_type == '64-QAM':
        # 64-QAM: تكوين شبكة 8x8 لنقاط التعديل
        symbols = (np.random.randint(0, 8, input_data.shape) - 3.5) + \
                  1j * (np.random.randint(0, 8, input_data.shape) - 3.5)
    elif modulation_type == '16-QAM':
        # 16-QAM: تكوين شبكة 4x4 لنقاط التعديل
        symbols = (np.random.randint(0, 4, input_data.shape) - 1.5) + \
                  1j * (np.random.randint(0, 4, input_data.shape) - 1.5)
    elif modulation_type == 'QPSK':
        # QPSK: أربع نقاط على الدائرة الوحدة
        symbols = (np.random.randint(0, 2, input_data.shape) * 2 - 1) + \
                  1j * (np.random.randint(0, 2, input_data.shape) * 2 - 1)
    else:
        raise ValueError("Unsupported modulation type")
        
    # Normalize to ensure constant average power
    return symbols * np.sqrt(1 / np.mean(np.abs(symbols)**2))
    
def apply_acm_and_channel(input_data, snr_db, channel):
    # Step 1: Apply ACM based on SNR
    modulation, coding_rate = adaptive_coding_modulation(snr_db)

    # Step 2: Simulate modulation more realistically
    modulated_data = modulate(input_data, modulation)

    # Step 3: Apply channel fading
    faded_data = channel(modulated_data)

    # Step 4: Apply coding rate
    coded_data = faded_data * coding_rate
    return coded_data


class Rayleigh_Channel(nn.Module):
    def __init__(self, snr_db=None, scale_factor=1.0, sigma=1.0):
        super(Rayleigh_Channel, self).__init__()
        self.scale_factor = scale_factor
        self.sigma = sigma
        self.snr_db = snr_db  # حفظ قيمة SNR

    def forward(self, x):
        if isinstance(x, np.ndarray):
            # إنشاء مصفوفة عشوائية باستخدام numpy بنفس شكل x
            fading = np.sqrt(-2 * self.sigma**2 * np.log(1 - np.random.rand(*x.shape)))
            signal = x * fading * self.scale_factor * self.sigma
        else:
            # استخدام الدالة الأصلية torch.rand_like إذا كان x من نوع Tensor
            fading = torch.sqrt(-2 * self.sigma**2 * torch.log(1 - torch.rand_like(x)))
            signal = x * fading * self.scale_factor * self.sigma

        # إذا تم توفير قيمة SNR، قم بتعديل الإشارة لتتناسب معها
        if self.snr_db is not None:
            snr_linear = 10 ** (self.snr_db / 10)
            power_signal = torch.mean(signal ** 2)
            noise_variance = power_signal / snr_linear
            noise = torch.sqrt(noise_variance) * torch.randn_like(signal)
            signal = signal + noise

        return signal

# مثال الاستخدام
# channel = Rayleigh_Channel(snr_db=args.snr_db).to(device)
# حساب متوسط القيم
class AverageMeter():
  def __init__(self, name):
      self.reset()
      self.name = name

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

  def __repr__(self):
      return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"

# إعداد مجموعة البيانات
class ImageDataset():
  def __init__(self, data_dir, transform=None):
      self.data_dir = data_dir
      self.transform = transform
      self.image_paths = [os.path.join(data_dir, img) for img in sorted(os.listdir(data_dir)) if img.endswith((".jpg", ".png"))]

  def __len__(self):
      return len(self.image_paths)

  def __getitem__(self, idx):
      img_path = self.image_paths[idx]
      image = Image.open(img_path).convert('RGB')
      if self.transform:
          image = self.transform(image)
      return image, img_path

# إعداد التحويلات
def get_transformation():
  return transforms.Compose([
      transforms.Resize((512, 512)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

# حساب خسارة LPIPS
def calc_lpips_loss(im1, im2, percept):
  img_gen_resize = F.adaptive_avg_pool2d(im1, (256, 256))
  target_img_tensor_resize = F.adaptive_avg_pool2d(im2, (256, 256))
  p_loss = percept(img_gen_resize, target_img_tensor_resize).mean()
  return p_loss

# إعداد المحسنات
def setup_optimizers(generator, discriminator):
  optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
  optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
  return optimizer_G, optimizer_D
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


# إعداد النموذج
def setup_models(device, latent_dim):
  generator = Generator(latent_dim).to(device)
  discriminator = Discriminator().to(device)
  vae = VAE(latent_dim).to(device)  # Pass latent_dim here
  return generator, discriminator, vae

# إعداد البيانات
def setup_data_loader():
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
  return DataLoader(dataset, batch_size=64, shuffle=True)

# إعداد السجل
def init_logger(args):
  t = time.strftime("%m_%d_%H:%M:%S", time.localtime())
  logger = get_logger(f"results/log/{t}-{args.snr_db}db.log")
  logger.info(args)
  return logger
def optimize_latent(args, g_ema, target_img_tensor, batch_size):

    noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    for noise in noises:
        noise.requires_grad = False
    # initialization
    with torch.no_grad():
        noise_sample = torch.randn(10000, 512, device=device)
        latent_mean = g_ema.style(noise_sample).mean(0)
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    # Channel
    if args.no_noises:
        optimizer = optim.Adam([latent_in], lr=args.lr)
    else:
        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    latent_path = [latent_in.detach().clone()]
    pbar = tqdm(range(args.step))
    latent_in.requires_grad = True
    for i in pbar:
        optimizer.zero_grad()
        optimizer.param_groups[0]['lr'] = get_lr(float(i)/args.step, args.lr)
        img_gen, _ = g_ema([channel(p_norm(latent_in, dim=(1, 2)))],
                           input_is_latent=True, randomize_noise=False, noise=None)

        # VGG loss
        p_loss = calc_lpips_loss(img_gen, target_img_tensor, percept)
        # L1_loss
        l1_loss = F.mse_loss(img_gen, target_img_tensor)
        # ssim_loss
        ssim_loss = 1 - ms_ssim(img_gen.clip(0, 1)*0.5+0.5,
                             target_img_tensor*0.5+0.5, data_range=1, size_average=True)
        if args.w_plus == True:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.unsqueeze(
                0).repeat(latent_in.size(0), g_ema.n_latent, 1))
        else:
            latent_mean_loss = F.mse_loss(
                latent_in, latent_mean.repeat(latent_in.size(0), 1))

        # main loss function
        loss = (
            p_loss * args.lambda_lpips +
            ssim_loss * args.lambda_ssim +
            l1_loss * args.lambda_l1 +
            latent_mean_loss * args.lambda_mean
        )
        pbar.set_description(
            f' ssim_loss: {ssim_loss.item():.4f} L1 loss: {l1_loss.item():.4f} VGG loss: {p_loss}')

        loss.backward()
        optimizer.step()

        # noise_normalize_(noises)
        latent_path.append(latent_in.detach().clone())

    return latent_path, noises

# الدالة الرئيسية
if __name__ == '__main__':
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  # إعداد المحلل
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', type=str, default='pretrained/CelebAMask-HQ-512x512.pt')
  parser.add_argument('--outdir', type=str, default='results/inversion')
  parser.add_argument('--dataset', default="./data/examples")
  parser.add_argument('--size', type=int, default=512)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--no_noises', type=lambda x: x.lower() in ['false', '0'], default=True)
  parser.add_argument('--w_plus', type=lambda x: x.lower() in ['true', '1'], default=True)
  parser.add_argument('--save_steps', type=lambda x: x.lower() in ['true', '1'], default=False)
  parser.add_argument('--truncation', type=float, default=1)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--step', type=int, default=300)
  parser.add_argument('--snr_db', type=int, default=15, help='snr in db')
  parser.add_argument('--noise_regularize', type=float, default=10)
  # parser.add_argument('--lambda_lpips', type=float, default=1.0, help='Weight for LPIPS loss')
  # parser.add_argument('--lambda_ssim', type=float, default=1.0, help='Weight for SSIM loss')
  parser.add_argument('--lambda_l1', type=float, default=1.0, help='Weight for L1 loss')
  parser.add_argument('--lambda_ssim', type=float, default=1.0, help='Weight for SSIM loss')
  # parser.add_argument('--lambda_l1', type=float, default=1.0, help='Weight for L1 loss')
  # parser.add_argument('--lambda_lpips', type=float, default=1.0, help='Weight for LPIPS loss')
  parser.add_argument('--lambda_lpips', type=float, default=1.0, help='Weight for LPIPS loss')
  # parser.add_argument('--lambda_ssim', type=float, default=1.0, help='Weight for SSIM loss')
  # parser.add_argument('--lambda_l1', type=float, default=1.0, help='Weight for L1 loss')
  # parser.add_argument('--lambda_mean', type=float, default=1.0, help='Weight for mean loss')
  parser.add_argument('--lambda_mean', type=float, default=1.0, help='Weight for the latent mean loss')
  # parser.add_argument('--lambda_lpips', type=float, default=1.0, help='Weight for LPIPS loss')
  # parser.add_argument('--lambda_ssim', type=float, default=1.0, help='Weight for SSIM loss')
  # parser.add_argument('--lambda_l1', type=float, default=1.0, help='Weight for L1 loss')
  # parser.add_argument('--lambda_mean', type=float, default=1.0, help='Weight for mean loss')  # Ensure this is defined only once
  
  # تعيين البذور العشوائية

  torch.manual_seed(42)
  seed = 42
  torch.cuda.manual_seed_all(seed)
  random.seed(seed)
  np.random.seed(seed)

  args = parser.parse_args()
  data_directory = '/content/GAN_SeCom/data'
  for root, dirs, files in os.walk(data_directory):
    print(f"Checking directory: {root}")
    for file in files:
        print(f"Found file: {file}")  # استبدل هذا بمسار مجلد الصور الخاص بك
  data_loader = load_data(data_directory)
  model_file_path = 'model.pth'
  train_and_save_model(data_directory, model_file_path, num_epochs=100, batch_size=32)
  results_dir = 'results'
  os.makedirs(results_dir, exist_ok=True)  # إنشاء المجلد إذا لم يكن موجودًا
  os.rename(model_file_path, os.path.join(results_dir, model_file_path))
  print(f"Model moved to {os.path.join(results_dir, model_file_path)}")

  # إعداد المجلدات
  args.outdir = os.path.join(args.outdir, str(args.snr_db) + "dB")
  if os.path.exists(args.outdir):
      shutil.rmtree(args.outdir)
  os.makedirs(os.path.join(args.outdir, 'recon'), exist_ok=True)
  if args.save_steps:
      os.makedirs(os.path.join(args.outdir, 'steps'), exist_ok=True)
  os.makedirs(os.path.join(args.outdir, 'latent'), exist_ok=True)
  if not args.no_noises:
      os.makedirs(os.path.join(args.outdir, 'noise'), exist_ok=True)
  if not os.path.exists(data_directory):
      print(f"Directory {data_directory} does not exist.")
  else:
      # تحقق من وجود ملفات الصور
    for root, dirs, files in os.walk(data_directory):
          # تجاهل المجلدات غير المرغوب فيها
        if '.ipynb_checkpoints' in dirs:
            dirs.remove('.ipynb_checkpoints')  # تجاهل هذا المجلد
        print(f"Checking directory: {root}")
        for file in files:
            print(f"Found file: {file}")  # طباعة أسماء الملفات
    try:
          batch_size = 32  # يمكنك تغيير حجم الدفعة حسب الحاجة
          data_loader = load_data(data_directory, batch_size)
          print(f"Loaded {len(data_loader.dataset)} images from {data_directory}.")
    except Exception as e:
          print(f"Error loading data: {e}")
    channel = Channel(noise_power=0.01)  # تقليل الضوضاء
    ris = RIS(num_elements=10)
    beamforming = Beamforming(num_antennas=4)
    swipt = SWIPT(capacity=100)
    diffusion_model = DiffusionModel(num_layers=5)
    rsma = RSMA(num_users=8)
    dynamic_power_control = DynamicPowerControl(max_power=100)
    time_aware_communication = TimeAwareCommunication(latency=10)
    semantic_accuracy = SemanticAccuracy(accuracy=0.9)
    privacy_protection = PrivacyProtection(protection_level=0.8)

    # إنشاء إشارة عشوائية
    signal = np.random.rand(100)

    # تمرير الإشارة عبر المكونات المختلفة
    received_signal = channel.transmit(signal)
    print("Received Signal:", received_signal)  # طباعة الإشارة المستلمة
    adapted_signal = ris.adapt(received_signal)
    print("Adapted Signal:", adapted_signal)  # طباعة الإشارة المعدلة
    steered_signal = beamforming.steer(adapted_signal)
    print("Steered Signal:", steered_signal)  # طباعة الإشارة الموجهة
    stored_signal = swipt.store(steered_signal)
    print("Stored Signal:", stored_signal)  # طباعة الإشارة المخزنة
    diffused_signal = diffusion_model.diffuse(stored_signal)
    print("Diffused Signal:", diffused_signal)  # طباعة الإشارة المنتشرة
    allocated_signal = rsma.allocate(diffused_signal)
    print("Allocated Signal:", allocated_signal)  # طباعة الإشارة المخصصة
    controlled_signal = dynamic_power_control.control(allocated_signal)
    print("Controlled Signal:", controlled_signal)  # طباعة الإشارة المتحكم فيها
    communicated_signal = time_aware_communication.communicate(controlled_signal)
    print("Communicated Signal:", communicated_signal)  # طباعة الإشارة المتبادلة
    encoded_signal = semantic_accuracy.encode(communicated_signal)
    print("Encoded Signal:", encoded_signal)  # طباعة الإشارة المشفرة
    protected_signal = privacy_protection.protect(encoded_signal)
    print("Protected Signal:", protected_signal)  # طباعة الإشارة المحمية

    # التحقق من النطاق
    assert np.all(protected_signal >= 0) and np.all(protected_signal <= 1), "Protected signal values are out of range!"

    # Created/Modified files during execution:
    print("None")
    # إعداد النماذج
    latent_dim=100
    generator, discriminator, vae = setup_models(device, latent_dim)
    optimizer_G, optimizer_D = setup_optimizers(generator, discriminator)
    dataloader = setup_data_loader()

    # تدريب نموذج GAN
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.view(images.size(0), -1).to(device)  # تسطيح الصور
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            # تدريب المميز
            optimizer_D.zero_grad()
            outputs = discriminator(images)
            d_loss_real = F.binary_cross_entropy(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(images.size(0), 100).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = F.binary_cross_entropy(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_D.step()

            # تدريب المولد
            optimizer_G.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = F.binary_cross_entropy(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss_real.item() + d_loss_fake.item():.4f}, g_loss: {g_loss.item():.4f}')

    # إعداد VAE
    optimizer_VAE = optim.Adam(vae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # تدريب نموذج VAE
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.view(images.size(0), -1).to(device)
            optimizer_VAE.zero_grad()
            reconstructed, mu, logvar = vae(images)
            loss = criterion(reconstructed, images) + 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
            loss.backward()
            optimizer_VAE.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], VAE Loss: {loss.item():.4f}')
    t = time.strftime("%m_%d_%H:%M:%S", time.localtime())
    # إعداد السجل
    logger = get_logger(
            f"results/log/{t}-{args.snr_db}db.log")
    logger.info(args)

    logger = init_logger(args)
    logger.info("Loading model ...")
    ckpt = torch.load(args.ckpt)
    g_ema = make_model(ckpt['args']).to(device)
    g_ema.eval()
    g_ema.load_state_dict(ckpt['g_ema'])
    percept = lpips.LPIPS(net_type='vgg').to(device)

    # إعداد المميز
    discriminator = DualBranchDiscriminator(args.size, args.size, img_dim=3, seg_dim=13, channel_multiplier=2).to(device)
    discriminator.load_state_dict(ckpt['d'])
    discriminator.eval()

    # إعداد القنوات
    p_norm = PowerNormalize(t_pow=1).to(device)
    channel = Rayleigh_Channel(snr_db=args.snr_db).to(device)
    channel.cuda()

    # إعداد مجموعة البيانات للاختبار
    transform = get_transformation(args)
    psnrs = []
    ms_ssims = []
    totlal_lpips = []
    nums = []
   
    test_dataset = ImageDataset(args.dataset, transform=transform)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=False)
    diffusion_model = create_gaussian_diffusion(
      steps=1000,
      noise_schedule="linear",
      learn_sigma=True,
      predict_xstart=True, )
    # حساب المقاييس
    iter_psnr = AverageMeter('Iter PSNR')
    iter_msssim = AverageMeter('MS-SSIM')
    iter_lpips = AverageMeter('LPIPS')

    for batch_idx, (images, path) in enumerate(data_loader):
        images = images.to(device)
        target_img_tensor = images
        latent_path, noises = optimize_latent(args, g_ema, images, images.shape[0])
        with torch.no_grad():
            latent = latent_path[-1]
            latent = channel(p_norm(latent_path[-1], dim=(1, 2)))
            img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, noise=None)
            lpips_img = calc_lpips_loss(img_gen, target_img_tensor, percept)
            img_y = img_gen.clamp(-1, 1) * 0.5 + 0.5
            target_img_tensor = target_img_tensor * 0.5 + 0.5
            psnr_img = piq.psnr(target_img_tensor, img_y)
            ssim_img = ms_ssim(target_img_tensor, img_y, data_range=1)

            # تحديث المقاييس
            iter_psnr.update(psnr_img, images.size(0))
            iter_msssim.update(ssim_img, images.size(0))
            iter_lpips.update(lpips_img, images.size(0))

            # حفظ الصور
            imgs = tensor2image(img_gen)
            for i in range(img_gen.shape[0]):
                img_path = os.path.join(args.outdir, 'recon/', path[i][-9:])
                imwrite(img_path, imgs[i])

    # تسجيل النتائج
    logger.info(f"Avg PSNR: {iter_psnr.avg}")
    logger.info(f"Avg MS-SSIM: {iter_msssim.avg}")
    logger.info(f"Avg LPIPS: {iter_lpips.avg}")

