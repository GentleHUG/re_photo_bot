if __name__ == "__main__":
    from telebot.async_telebot import AsyncTeleBot
    from telebot import types
    import os
    import torch
    import tensorflow as tf
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import pytorch_lightning as L

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import cv2
    matplotlib.use('agg')

    from torch.utils.data import DataLoader
    from pytorch_lightning.utilities import CombinedLoader
    from torchvision.io import read_image

    # Import custom classes
    from custom_classes import CustomTransform, CustomDataset, CycleGAN

    # Set seed for reproducibility
    _ = L.seed_everything(0, workers=True)
    print("num_workers", os.cpu_count(), "pin_memory", torch.cuda.is_available())

    from PIL import Image

    TOKEN = os.getenv("TOKEN")
    model_path = './model.pth'
    frame_path = "./frame1.png"
    def resize_frame_to_fit_photo(frame, photo):
        # Определяем размеры фото
        photo_width, photo_height = photo.size

        # Изменяем размер рамки так, чтобы прозрачная область соответствовала размеру фото
        # Здесь предполагаем, что прозрачная область рамки занимает большую часть изображения.
        frame_width, frame_height = frame.size
        scale_width = photo_width / frame_width
        scale_height = photo_height / frame_height

        # Масштабируем рамку
        frame_resized = frame.resize((int(frame_width * scale_width), int(frame_height * scale_height)),
                                     Image.Resampling.LANCZOS)

        return frame_resized


    # Открываем изображения

    def frame (frame_path, photo_path, new_src):
        frame = Image.open(frame_path).convert("RGBA")
        photo = Image.open(photo_path).convert("RGBA")

        # Изменяем размер рамки
        frame_resized = resize_frame_to_fit_photo(frame, photo)

        # Создаем новое изображение с размерами измененной рамки
        combined = Image.new("RGBA", frame_resized.size)

        # Рассчитываем координаты для вставки фото в центр измененной рамки
        frame_resized_width, frame_resized_height = frame_resized.size
        photo_width, photo_height = photo.size

        offset_x = (frame_resized_width - photo_width) // 2
        offset_y = (frame_resized_height - photo_height) // 2

        # Наложение фото на новое изображение с учетом смещения
        combined.paste(photo, (offset_x, offset_y), photo)

        # Наложение измененной рамки на новое изображение
        combined.paste(frame_resized, (0, 0), frame_resized)
        new_img = combined.convert('RGB')

        # Сохранение результата
        new_img.save(new_src)
    class CustomDataModule(L.LightningDataModule):
        def __init__(self, monet_dir, photo_dir, loader_config, sample_size, batch_size):
            super().__init__()
            self.loader_config = loader_config
            self.sample_size = sample_size
            self.batch_size = batch_size
            self.monet_filenames = monet_dir
            self.photo_filenames = photo_dir
            self.transform = CustomTransform()

        def setup(self, stage):
            if stage == "fit":
                self.train_monet = CustomDataset(self.monet_filenames, self.transform, stage)
                self.train_photo = CustomDataset(self.photo_filenames, self.transform, stage)

            if stage in ["fit", "test", "predict"]:
                self.valid_photo = CustomDataset(self.photo_filenames, self.transform, None)

        def train_dataloader(self):
            loader_config = {
                "shuffle": True,
                "drop_last": True,
                "batch_size": self.batch_size,
                **self.loader_config,
            }
            loader_monet = DataLoader(self.train_monet, **loader_config)
            loader_photo = DataLoader(self.train_photo, **loader_config)
            loaders = {"monet": loader_monet, "photo": loader_photo}
            return CombinedLoader(loaders, mode="max_size_cycle")

        def val_dataloader(self):
            return DataLoader(self.valid_photo, batch_size=self.sample_size, **self.loader_config)

        def test_dataloader(self):
            return self.val_dataloader()

        def predict_dataloader(self):
            return DataLoader(self.valid_photo, batch_size=self.batch_size, **self.loader_config)


    DEBUG = False

    DM_CONFIG = {
        "loader_config": {
            "num_workers": os.cpu_count(),
            "pin_memory": torch.cuda.is_available(),
        },
        "sample_size": 5,
        "batch_size": 5 if not DEBUG else 1,
    }

    # Continue defining other classes and training logic here...
    # Ensure all necessary imports and definitions are included

    # At the end of your script, start the training

    # Define model and training logic here...

    MODEL_CONFIG = {
        # the type of generator, and the number of residual blocks if ResNet generator is used
        "gen_name": "unet",  # types: 'unet', 'resnet'
        "num_resblocks": 6,
        # the number of filters in the first layer for the generators and discriminators
        "hid_channels": 64,
        # using DeepSpeed's FusedAdam (currently GPU only) is slightly faster
        "optimizer": torch.optim.Adam,
        # the learning rate and beta parameters for the Adam optimizer
        "lr": 2e-4,
        "betas": (0.5, 0.999),
        # the weights used in the identity loss and cycle loss
        "lambda_idt": 0.5,
        "lambda_cycle": (10, 10),  # (MPM direction, PMP direction)
        # the size of the buffer that stores previously generated images
        "buffer_size": 100,
        # the number of epochs for training
        "num_epochs": 26 if not DEBUG else 2,
        # the number of epochs before starting the learning rate decay
        "decay_epochs": 26 if not DEBUG else 1,
    }

    TRAIN_CONFIG = {
        "accelerator": "cpu",

        # train on 16-bit precision
        "precision": 32,

        # train on single GPU
        "devices": 1,

        # save checkpoint only for last epoch by default
        "enable_checkpointing": True,

        # disable logging for simplicity
        "logger": False,

        # the number of epochs for training (we limit the number of train/predict batches during debugging)
        "max_epochs": MODEL_CONFIG["num_epochs"],
        "limit_train_batches": 1.0 if not DEBUG else 2,
        "limit_predict_batches": 1.0 if not DEBUG else 5,

        # the maximum amount of time for training, in case we exceed run-time of 5 hours
        "max_time": {"hours": 4, "minutes": 55},

        # use a small subset of photos for validation/testing (we limit here for flexibility)
        "limit_val_batches": 1,
        "limit_test_batches": 5,

        # disable sanity check before starting the training routine
        "num_sanity_val_steps": 0,

        # the frequency to visualize the progress of adding Monet style
        "check_val_every_n_epoch": 6 if not DEBUG else 1,
    }

    model = CycleGAN(**MODEL_CONFIG)
    model.load_state_dict(torch.load(model_path))
    trainer = L.Trainer(**TRAIN_CONFIG)

    def save_img(img_tensor, name):

        img_tensor = img_tensor*0.5+0.5
        plt.matshow(img_tensor)
        plt.axis("off")
        plt.savefig(name, bbox_inches='tight', pad_inches=0)

    def re_photo(src, new_src):
        datamodule = CustomDataModule(
            monet_dir=[src], photo_dir=[src],
            **DM_CONFIG
        )
        predictions = trainer.predict(model, datamodule=datamodule)
        model.to("cpu")
        img = read_image(src)
        # Apply the transformation on the input image
        p = np.squeeze(predictions[0])

        imge = tf.image.resize(p.permute(1, 2, 0), list(img.shape)[1:])
        alpha = 0.5
        # print(np.array(img.permute(1,2,0)).astype(float), np.array(imge).shape)
        combined_image = cv2.addWeighted(np.array(img.permute(1,2,0)).astype(float), 1, np.array(imge).astype(float), 0, 0)

        save_img(imge, new_src)


    bot = AsyncTeleBot(TOKEN)

    # Глобальная переменная для хранения текущего действия пользователя
    current_action = {}


    # Стартовая команда /start
    @bot.message_handler(commands=['start'])
    async def start(message):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("👋 Поздороваться")
        markup.add(btn1)
        await bot.send_message(message.chat.id,f'Привет, {message.from_user.username}!\nЭтот бот поможет тебе обработать фотографии', reply_markup=markup)


    # Обработка текстовых сообщений
    @bot.message_handler(content_types=['text'])
    async def get_text_messages(message):
        if message.text == '👋 Поздороваться':
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton('Добавить рамку')
            btn2 = types.KeyboardButton('Сделать фото пленочным')
            markup.add(btn1, btn2)
            await bot.send_message(message.from_user.id, 'Выбери, что тебя интересует', reply_markup=markup)
        elif message.text == 'Сделать фото пленочным':
            current_action[message.from_user.id] = 'film_effect'
            await bot.send_message(message.from_user.id, 'Отправь фотографию, которую нужно сделать пленочной',
                             parse_mode='Markdown')
        elif message.text == 'Добавить рамку':
            current_action[message.from_user.id] = 'add_frame'
            await bot.send_message(message.from_user.id, 'Отправь фотографию, к которой нужно добавить рамку',
                             parse_mode='Markdown')


    # Обработка фотографий
    @bot.message_handler(content_types=['photo'])
    async def handle_photo(message):
        user_id = message.from_user.id
        if user_id in current_action:

            file_info = await bot.get_file(message.photo[-1].file_id)
            downloaded_file = await bot.download_file(file_info.file_path)
            src = './' + file_info.file_path
            src = src[:2] + src[9:]
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

            if current_action[user_id] == 'film_effect':
                await bot.send_message(user_id, 'Перепленочитываю')
                re_photo(src, src)
                await bot.send_photo(user_id, photo=open(src, 'rb'))
            elif current_action[user_id] == 'add_frame':
                await bot.send_message(user_id, 'Ищу свой Instax')
                frame(frame_path, src, src)
                await bot.send_photo(user_id, photo=open(src, 'rb'))

            # Сброс текущего действия
            current_action.pop(user_id)
        else:
            await bot.send_message(user_id, 'Пожалуйста, выберите действие перед отправкой фотографии.')


    import asyncio

    asyncio.run(bot.infinity_polling())