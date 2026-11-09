# Предобучение S AI LLM 1.0

[Отчёт и результаты](REPORT.md). Исходный `CopyLLM` содержит 1 033 003 011 параметров. Файлы весов загружаются с `weights_only=True`.

## Colab

Откройте [ноутбук T4](S_AI_LLM_1_0_Pretraining_T4.ipynb), выберите Tesla T4 и выполняйте ячейки по порядку. В режиме отдельных файлов загрузите в `/content` исходный `s_ai_llm_1_0_untrained_expanded.pt` и, при наличии, `sai1b_wikipedia_pilot_10m.tar.gz`. Папка результатов — `/content/sai1b-results`. Скачайте состояние и метрики до завершения сессии. Для работы через Google Drive включите `USE_DRIVE` в первой кодовой ячейке. Тогда результаты будут в `MyDrive/S_AI_Training/S_AI_LLM_1_0/pretraining_v3_20261108`.

Пилот сохраняет состояние после шага 12 и продолжает до шага 24. Для возобновления в новом сеансе выполните ячейки настройки и корпуса, затем ячейку возобновления. В режиме отдельных файлов сначала верните сохранённый `training-state.pt` в `/content/sai1b-results/t4-pilot`.

## Продолжение после шага 24 на T4

[Профиль продолжения](t4_continuation.json) задаёт 512 новых шагов, 524 288 позиций поровну на RU и EN и обновление всех 20 блоков. Состав validation и test тот же, что в пилоте при использовании того же корпуса. Новый seed меняет последовательность тренировочных окон. Новый этап начинает собственное расписание скорости обучения и сохраняет состояние каждые 128 шагов.

В Colab выполните ячейки настройки и подготовки корпуса. В терминале перейдите в корень актуальной ветки репозитория. Для `PILOT` используйте локальную копию `t4-pilot/training-state.pt`, для `CORPUS` — каталог распакованного пилотного корпуса, для `RUN` — новый каталог на диске с сохранением между сеансами.

```bash
PILOT=/path/to/local/t4-step24.pt
CORPUS=/path/to/wiki-pilot
RUN=/path/to/persistent/t4-continuation
printf '0f5f6547d27e1482571f8c8b5defae08b14da072d2dfa720b0c049100cce9044  %s\n' "$PILOT" | sha256sum --check &&
  python -m training_1b.pretrain --profile training_1b/t4_continuation.json \
  --corpus "$CORPUS" --init "$PILOT" --output "$RUN" --stop-after 128
```

После остановки на шаге 128 или прерывания возьмите локальную копию последнего `training-state.pt` из нового каталога. Для возобновления используйте тот же профиль, корпус и версии среды.

```bash
RESUME=/path/to/local/t4-continuation-state.pt
python -m training_1b.pretrain --profile training_1b/t4_continuation.json \
  --corpus "$CORPUS" --resume "$RESUME" --output "$RUN"

FINAL=/path/to/local/t4-continuation-final.pt
python -m training_1b.evaluate --checkpoint "$FINAL" \
  --output "$RUN/answers.json" --device cuda --precision fp16 --max-tokens 32
```

Повторите проверку с `--vocabulary-only --output "$RUN/answers-vocabulary.json"`, чтобы отдельно увидеть, что даёт обученная словарная проекция. Обычный вызов оценивает штатную генерацию CopyLLM.

## Проверочный запуск на Selectel PRO 6000 96 ГБ

На уже запущенном сервере Linux перенесите в закрытый каталог чекпойнт шага 24 и архив `sai1b_wikipedia_pilot_10m.tar.gz`. Убедитесь, что драйвер видит GPU, и установите PyTorch с CUDA 12.8 в отдельное виртуальное окружение. [GPU в Selectel](https://docs.selectel.ru/cloud-servers/create/gpus/) · [Сборки PyTorch](https://pytorch.org/get-started/previous-versions/)

```bash
nvidia-smi
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install tokenizers==0.21.4 filelock
python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(including_emulation=False)
sample = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
assert (sample @ sample)[0, 0].item() == 256
print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)
PY
```

Из корня рабочей ветки укажите проверенный локальный файл `PILOT`, распакованный `CORPUS` и новый постоянный каталог `RUN` по образцу выше. Начните с 16 шагов BF16 на существующем профиле. Это 1 048 576 позиций с записью состояния в конце отрезка.

```bash
printf '0f5f6547d27e1482571f8c8b5defae08b14da072d2dfa720b0c049100cce9044  %s\n' "$PILOT" | sha256sum --check &&
  python -m training_1b.pretrain --profile training_1b/a100_h100.json \
    --corpus "$CORPUS" --init "$PILOT" --output "$RUN" --stop-after 16
```

По `metrics.json` проверьте токены в секунду, время шагов, пик памяти и validation loss обоих языков. Для повторного запуска с тем же профилем и корпусом используйте `--resume "$RUN/training-state.pt"` и более поздний `--stop-after`. При переходе на больший корпус начните отдельный этап через `--init`.

## A100 или H100

```bash
git clone --branch codex/sai-llm-1b-pretraining https://github.com/thepeshkov7/S-AI-LLM.git
cd S-AI-LLM
python -m pip install -r requirements.txt filelock pyarrow==20.0.0
BASE=/path/to/s_ai_llm_1_0_untrained_expanded.pt

python -m training_1b.corpus --checkpoint "$BASE" --cache corpus/cache \
  --output corpus/wiki-100m --wiki-shards 2 --limit-tokens 50000000

python -m training_1b.pretrain --profile training_1b/a100_h100.json \
  --corpus corpus/wiki-100m --init "$BASE" --output runs/a100 --stop-after 200

python -m training_1b.pretrain --profile training_1b/a100_h100.json \
  --corpus corpus/wiki-100m --resume runs/a100/training-state.pt \
  --output runs/a100 --stop-after 500
```

Измерьте `train_tokens_per_second`, `peak_vram_allocated_gib` и validation каждого языка. Возобновление сверяет корпус, профиль, версии Torch и оптимизатора, аппаратную платформу и сохранённое состояние. Для новой фазы с другим профилем используйте `--init` и отдельный каталог. Полное состояние 1B с AdamW может занять около 11,6 ГиБ. При атомарной записи понадобится место для двух состояний.

Для T4 используется `t4_pilot.json` и `bitsandbytes==0.48.2`. `last_blocks` определяет число размороженных блоков. `objective: copy` включает градиенты указателя копирования. Полный срез Wikipedia доступен через `--wiki-shards 0 --limit-tokens 0`.

Корпус диалогов и задач можно воспроизвести командой `python -m training_1b.instructions --checkpoint "$BASE" --cache corpus/cache --output corpus/instructions`. Перед дальнейшим обучением используйте результат пересборки. Сборщик исключает шаблон `pencil-boxes-3-4-13` с некорректным числом использованных карандашей, попавший в первый архив.

## Восстановление CPU-пилота

На момент публикации в Drive были доступны 12 из 14 частей. Для восстановления с имеющимися файлами заново воспроизведите CPU-пилот по профилю ниже. Когда доступны все части, сложите все 14 частей с номерами 00–13 в один каталог и объедините их. Проверьте SHA-256 полученного файла по [метрикам](results/cpu-pilot.json).

```bash
cat sai1b_cpu_step8.pt.part-* > sai1b_cpu_step8.pt
sha256sum sai1b_cpu_step8.pt
```

Из корня репозитория восстановите файл из исходных весов и компактного состояния.

```bash
python -m training_1b.portable restore --base "$BASE" \
  --input /path/to/sai1b_cpu_step8.pt --output runs/cpu-step8-restored.pt
python -m training_1b.evaluate --checkpoint runs/cpu-step8-restored.pt \
  --output runs/cpu-step8-answers.json --device cpu --precision bf16 --max-tokens 24
```

Для воспроизведения исходного CPU-пилота возьмите `cpu_pilot.json`, `--device cpu` и `--stop-after 4`, затем выполните `--resume` с тем же профилем. Для нового этапа после шага 8 используйте `--init` и отдельный профиль.

При распространении данных Wikipedia сохраняйте `source-manifest.json` и `sources.jsonl.gz` с атрибуцией. Лицензии диалогов, инструкций и задач записаны в их манифесте.

## Продолжение S AI LLM 1.0 с шага 10 000

Профиль [step10000_recovery.json](step10000_recovery.json) загружает все 1 276 332 035 параметров CopyLLM и исходный словарь на 65 536 токенов из `step-010000.pt`. Первый запуск создаёт новый AdamW и расписание. Это продолжение исходных весов, а состояние оптимизатора старого `train_long.py` требует отдельной проверки его формата. Последующие запуски восстанавливают модель, оптимизатор, генераторы случайных чисел и позицию в расписании из `training-state.pt` этой стадии. Используйте отдельный каталог, чтобы сохранить исходный чекпойнт и предыдущие прогоны.

На каждом шаге модель читает 2 048 позиций, по 1 024 на язык при текущем профиле. Для RU и EN отдельно перемешиваются непересекающиеся окна по 256 позиций. Окна выбираются по всей длине каждого потока и используются максимум один раз за стадию. Хвост короче 256 токенов отбрасывается. До первого запуска можно изменить `ru_micro_batches` в копии профиля, чтобы задать другое соотношение. Профиль рассчитан на 1 024 новых шага и до 2 097 152 целевых токенов. Фактическое число целей меньше из-за маски на границах документов. Повторное использование данных из первых 10 000 шагов возможно.

Модель учится предсказывать EOS, а цель сразу после EOS исключается из cross-entropy. Внутри упакованного окна предыдущие документы остаются видимы через каузальное внимание. Градиент нормируется по фактическому числу целей после маскирования. Манифест и словарь сверяются до загрузки модели в GPU. Потоки проверяются на недопустимые ID, пустые документы, неверные границы и точные совпадения документов между train, validation и test.

В версии из репозитория causal SDPA и абсолютные позиции проверены тестами на изменение будущих токенов и на совпадение пошагового декодирования с полным проходом. SDPA использует стандартное масштабирование, а словарная cross-entropy вычисляется в FP32 даже под BF16. Остаточные соединения и нормализация сохранены. В локальном тесте возобновлённый прогон с маской EOS дал те же веса, что непрерывный. Файл серверного `train_long.py` и запуск на RTX 4090 требуют доступа к серверу. [Документация PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)

### Первая сессия после согласования бюджета

Выполняйте команды из корня обновлённого репозитория. При наличии старой копии ветки выполните `git pull --ff-only`. Профиль использует одну GPU с нативным BF16 и полный AdamW. Обычная RTX 4090 имеет 24 ГБ на карту. Указанные для сервера 48 ГБ нужно сверить по каждой карте, поскольку этот профиль работает на одном устройстве. Веса, градиенты и два момента AdamW занимают около 19 ГиБ без активаций и служебной памяти. Пиковое потребление на сервере предстоит измерить. Оставьте не менее 40 ГБ на локальном диске для состояний и временных файлов. [Характеристики RTX 4090](https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4090/)

```bash
PY=/root/S-AI-LLM/.venv/bin/python
DATA=/root/S-AI-Training/sai-1b-vocab65536
BASE="$DATA/long-run/checkpoints/step-010000.pt"
RUN="$DATA/recovery-step10000-v2"
PROFILE=training_1b/step10000_recovery.json

# Предлагаемый первый лимит — до 128 новых шагов и 15 минут на процесс.
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900s "$PY" -m training_1b.pretrain \
  --profile "$PROFILE" --corpus "$DATA/packed-v2" --output "$RUN" \
  --init "$BASE" --stop-after 128 --max-session-tokens 262144 --max-session-seconds 780
```

При достижении мягкого лимита обучение останавливается между шагами и сохраняет последний завершённый шаг. Время включает открытие корпуса, загрузку, обучение, штатную валидацию и сохранение. Разница в 120 секунд оставляет время на завершение работы. При принудительном завершении действителен последний успешно заменённый `training-state.pt`. Состояние сохраняется каждые 32 шага и при плановой остановке. При остановке до первого сохранения можно повторить команду с `--init`. События автоматически записываются в `training-log.jsonl`, сводка — в `metrics.json`. [GNU timeout](https://www.gnu.org/software/coreutils/manual/html_node/timeout-invocation.html)

Для возобновления возьмите номер стадии из последней строки `"event": "saved"`. В примере ниже она дошла до 128. Если номер другой, укажите фактическое значение. `--stop-after` задаёт итоговый шаг стадии. Токенный лимит считается для дополнительных шагов этой сессии.

```bash
SAVED=128
NEXT=$((SAVED + 128))
# NEXT должен быть не больше 1024.
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900s "$PY" -m training_1b.pretrain \
  --profile "$PROFILE" --corpus "$DATA/packed-v2" --output "$RUN" \
  --resume "$RUN/training-state.pt" --stop-after "$NEXT" \
  --max-session-tokens 262144 --max-session-seconds 780
```

При тарифе 180 ₽/ч 15 минут процесса соответствуют 45 ₽ пропорционально времени. Счёт провайдера зависит от длительности аренды и округления. Жёсткий лимит завершает процесс, а аренда управляется отдельно. Для пересчёта используйте измеренную скорость и добавляйте время проверки корпуса, валидации и записи состояний.

При замене корпуса или расписания создайте новый каталог и после согласования передайте предыдущий `training-state.pt` через `--init`. Новая стадия наследует веса и абсолютный номер шага и начинает собственное расписание с новым AdamW. Для обычного возобновления используйте `--resume` с тем же профилем и корпусом. Если новый корпус частично совпадает со старым, возможна повторная выборка этих данных.

### Генерация

Для исходного чекпойнта и новых состояний стадии генератор использует словарный выход CopyLLM, соответствующий цели обучения. По умолчанию он снижает вероятность недавних токенов продолжения и блокирует повтор трёхтокенных последовательностей до выбора top-k. Top-p применяется после top-k, frequency penalty пропорционален числу появлений токена в продолжении. EOS завершает генерацию. Связность текста зависит от весов модели.

```bash
"$PY" generate.py --checkpoint "$BASE" --device cuda \
  --prompt "Сегодня утром я проснулся и" --max-new-tokens 48 \
  --top-k 40 --top-p 0.95 --temperature 0.85 --repetition-penalty 1.2 \
  --frequency-penalty 0.1 --no-repeat-ngram-size 3
```

### Экзамен и условие перехода к диалогам

Сохранённые [ответы из исходного экзамена](results/step10000-exam-pasted.json) показывают 0/100 по предоставленной оценке. Настройки того прогона неизвестны. Для сопоставления checkpoint 10 000 с новым состоянием запустите один и тот же тест на обоих файлах. Команда сохраняет необработанные продолжения и настройки, отдельно считает RU и EN и позволяет внести оценки связности, грамматики, правильности, соблюдения инструкций и повторов. В `--suite prose` доступны восемь новых отрывков с эталонами для NLL и perplexity. В `--suite all` добавлены десять вопросов экзамена и шесть дополнительных задач на обобщение.

```bash
"$PY" -m training_1b.exam --checkpoint "$BASE" --device cuda \
  --suite all --output exam-step10000.json
"$PY" -m training_1b.exam --checkpoint "$RUN/training-state.pt" --device cuda \
  --suite all --output exam-recovery.json
"$PY" -m training_1b.exam --summarize exam-recovery.json
```

Сначала сравните связность обычного текста по обоим языкам и loss на одном и том же наборе. Одно удачное продолжение или падение loss недостаточно для перехода к диалогам. Когда базовая модель стабильно продолжает незнакомый текст в обоих языках, подготовьте отдельный набор диалогов, инструкций и арифметики с токенизатором выбранного состояния. Сборщик исключает известные вопросы экзамена и разделяет данные по группам. Обучение в этой стадии маскирует текст пользователя и предсказывает ответ и EOS. Запуск требует отдельного согласования.

```bash
# Только после оценки базового текста. Подготовка данных выполняется на CPU.
SFT_BASE="$RUN/training-state.pt"
"$PY" -m training_1b.instructions --checkpoint "$SFT_BASE" \
  --cache corpus/cache --output corpus/instructions-recovery
# После отдельного согласования SFT. Используйте новый путь для выходных весов.
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900s "$PY" -u tune_bilingual.py \
  --base "$SFT_BASE" --curriculum corpus/instructions-recovery \
  --output runs/sai-dialogue/candidate.pt --device cuda --bf16 \
  --gradient-checkpointing --reviewed-base-prose --batch-size 2 \
  --learning-rate 0.00002 --steps 128 --eval-every 16 --stop-after 32
```

### Расширение корпуса

Около 20,48 млн обработанных позиций — примерно 0,016 на параметр модели. Даже один проход по 134 млн токенов даёт около 0,1 токена на параметр. Это объём для проверки процесса и раннего прогресса. Для масштаба, в исследованиях вычислительно эффективного предобучения ориентир порядка 20 токенов на параметр соответствует примерно 25,5 млрд для S AI. Это исследовательский ориентир, а объём до первого полезного качества S AI определится по измерениям. [Hoffmann et al.](https://arxiv.org/abs/2203.15556)

1. Соберите промежуточную версию на 0,5–1 млрд токенов, начиная с примерно равных бюджетов RU и EN. Добавьте документальную и художественную прозу, собственные тексты с разрешением и статьи Wikimedia. Храните версии источников, исходные ссылки, лицензии и изменения. Для книг проверяйте статус конкретного произведения и перевода в применимой юрисдикции. [Условия Wikimedia](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [Условия Project Gutenberg](https://www.gutenberg.org/policy/license.html)
2. До упаковки очистите тексты от навигации, повторяющихся шаблонов, битой кодировки и специальных токенов. Удаляйте точные дубли по нормализованному тексту, затем группируйте близкие дубли по перекрывающимся фрагментам. Делите целые группы документов на train, validation и test до токенизации. Храните отдельный набор авторских отрывков и вопросы экзамена вне обучения.
3. Используйте исходный токенизатор. Если старый `packed-v2` не проходит проверки, пересоберите в новом каталоге с `python -m training_1b.corpus --checkpoint "$BASE" --cache corpus/cache --output corpus/packed-v3 --wiki-shards 0 --limit-tokens 0`. Для нескольких лицензированных источников `packed_corpus.py` принимает несколько `--source` и `--tokenizer-from` с экспортированным токенизатором чекпойнта. Для старого корпуса точная проверка документов не выявляет все близкие дубли и не восстанавливает историю того, какие тексты модель уже видела.
4. Оцените пропускную способность после первого разрешённого отрезка. Для 1 млрд токенов сценарии 1 000 и 6 000 токенов/с дают примерно 278 и 46 часов чистого обучения, или 50 000 и 8 300 ₽ при 180 ₽/ч. Это расчётные сценарии без времени на валидацию, сохранение и простой. Они не являются замером RTX 4090.
