from aiogram import *
from raytensor import RayTensor

bot = Bot(token="5953669002:AAF7F6LS7EKws7fkzDokmF6S727bbw2mtFg")
dp = Dispatcher(bot)


@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["Помощь"]
    keyboard.add(*buttons)
    await message.answer(
        'Добро Пожаловать в RayTensor!\nНажмите "Помощь" для получения инструкций',
        reply_markup=keyboard,
    )


@dp.message_handler(
    commands="xray", commands_ignore_caption=False, content_types=["photo"]
)
async def xray_image_scan(message: types.Message):
    await message.photo[-1].download("static/destination/xray.jpg")
    predict = RayTensor().xray_predict("static/destination/xray.jpg")
    await message.answer(f"С вероятностью {round(predict[2], 2)}% у Вас {predict[3]}")
    await message.answer(
        "Другие возможные диагнозы:\n"
        f"Вероятность пневмонии: {round(predict[1][2])}%\n"
        f"Вероятность COVID-19: {round(predict[1][0])}%\n"
        f"Вероятность туберкулёза: {round(predict[1][3])}%\n"
        f"С лёгкими всё в порядке: {round(predict[1][1])}%\n"
    )


@dp.message_handler(
    commands="ct", commands_ignore_caption=False, content_types=["photo"]
)
async def ct_image_scan(message: types.Message):
    await message.photo[-1].download("static/destination/ct.jpg")
    predict = RayTensor().ct_predict("static/destination/ct.jpg")
    await message.answer(f"С вероятностью {round(predict[2], 2)}% у Вас {predict[3]}")
    await message.answer(
        f"Другие возможные диагнозы:\n"
        f"Вероятность пневмонии: {round(predict[1][2])}%\n"
        f"Вероятность COVID-19: {round(predict[1][0])}%\n"
        f"С лёгкими всё в порядке: {round(predict[1][1])}%"
    )


@dp.message_handler(lambda message: message.text == "Помощь")
async def bot_help(message: types.Message):
    await message.answer(
        "Для сканирования снимков Вы должны отправить фото боту с подписями "
        "/ct - для КТ и /xray - для рентгена.",
    )


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
