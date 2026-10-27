"""Original study notes and small exercises for the released lesson 7, not homework solutions."""
from copy import deepcopy

LECTURE='lesson_7/lecture.ipynb'
HOMEWORK='lesson_7/homework.ipynb'
FAISS='https://github.com/facebookresearch/faiss/wiki/'


def extend(lessons, topics, questions, cards, diagrams, corrections):
    lessons.append(dict(id=7,title='Self-Supervised и Metric Learning',short='SSL и Metric Learning',tag='FEATURES & RETRIEVAL',
                        description='SimCLR, BYOL, DINO, эмбеддинги, face losses, retrieval и векторный поиск.',color='#ab8bd8'))
    bank={}

    def T(key,title,cells,definition,detail,formula,math,given,steps,trap,qa,path=LECTURE):
        source=dict(path=path,cells=cells)
        t=dict(id=key,lesson=7,title=title,definition=definition,detail=detail,formula=formula,
               mathNotes=math+'\n\n$$ '+formula+' $$',example=given,trap=trap,source=source,
               supplemental=True,diagrams=[],codeCards=[],
               walkthrough=dict(title=title+': по шагам',given=given,
                  steps=[dict(title=h,text=s) for h,s in zip(['Представляем вход','Вычисляем','Проверяем смысл'],steps)],
                  conclusion=trap,source=deepcopy(source)))
        topics.append(t);bank[key]=t
        for i,row in enumerate(qa,1):
            prompt,answer,*extra=row
            q=dict(id='l7-'+key+'-'+str(i),lesson=7,topic=key,collection='materials',type='open',level='reason',
                   prompt=prompt,answer=answer,example=given,source=deepcopy(source),
                   rubric=['Назван нужный механизм или определение','Объяснены шаги и результат примера','Указано условие применимости или ограничение'])
            if extra:
                kind,params=extra;q.pop('rubric');q.update(type=kind,level='apply',**params)
            questions.append(q)

    def N(value,calc,tol=1e-5):return ('number',dict(value=value,tolerance=tol,calculation=calc))
    def S(options,correct=0):return ('single',dict(options=options,correct=[correct]))

    T('ssl-simclr','SimCLR: views, encoder и projection head',[1,2],
      'SimCLR сближает две аугментированные версии одного изображения и противопоставляет им другие views батча.',
      'Для N изображений создают 2N views. Общий encoder f даёт признаки h, projection head g — векторы z для contrastive loss. Для anchor i другая view исходного изображения — positive, сам i исключается из softmax-знаменателя. Остальные 2N−2 views служат negatives, хотя среди них возможны семантически близкие объекты. Температура задаёт масштаб logits. Для downstream часто берут h до projection head. Аугментации определяют, к каким изменениям модель учится быть устойчивой: удаление маленького объекта crop-ом может разрушить смысл positive pair.',
      r'\ell_{i,j}=-\log\frac{\exp(\operatorname{cos}(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(\operatorname{cos}(z_i,z_k)/\tau)}',
      'z нормированы, τ>0. При N=3 у каждого anchor один positive и четыре negatives; знаменатель включает positive, но не self. Вычисляйте через cross_entropy/logsumexp, не вручную через большие exp.',
      'N=3 изображения → 6 views → similarity matrix 6×6. В строке anchor остаются 5 кандидатов: 1 positive и 4 negatives.',
      ['Из каждого из трёх исходников создаём по две views; используем один encoder с общими весами.',
       'Нормируем z, считаем z·zᵀ/τ и закрываем шесть диагональных self-logits.',
       'Target каждой строки — индекс второй view того же изображения; encoder h затем можно оценить linear probe.'],
      'Нельзя считать любую случайную аугментацию сохраняющей нужную семантику; большой batch также увеличивает число false negatives.',[
       ('Зачем SimCLR отдельный projection head, если на downstream его часто отбрасывают?','Head позволяет contrastive objective действовать в отдельном пространстве z, сохраняя в h полезные downstream-признаки. Encoder и head обучаются совместно.'),
       ('N=8 исходных изображений, по две views. Сколько negatives у одного anchor в SimCLR?', '16−2=14: исключаем anchor и его positive.',*N(14,r'2\cdot8-2=14')),
       ('Почему нельзя включать self-similarity на диагонали в число обычных кандидатов?','Модель могла бы выигрывать от сравнения представления с самим собой. Предсказать другую view — более содержательная задача.'),
       ('Какая пара является positive в исходном SimCLR?','Две views одного исходного изображения, а не любые два изображения одного неизвестного класса.',*S(['Две аугментации одного исходника','Два максимальных score в batch','Любые два одинаковых label, поданных loss','Всегда два соседних пикселя']))])

    T('ssl-byol','BYOL: predictor, stop-gradient и EMA',[3],
      'BYOL учит online-ветвь предсказывать представление другой view из медленно обновляемой target-ветви без явных negatives.',
      'Online encoder+projector дополняется predictor; target encoder+projector predictor не имеет. Вектор target отделяется от autograd. После optimizer step online-параметры смешиваются с target через EMA на месте, без разделения одного Parameter между ветвями. Loss симметризуют по двум направлениям views. Во время применения обычно нужен encoder, а не вся схема предобучения. Отсутствие negatives само по себе не объясняет устойчивость: важна вся обучающая асимметрия и recipe, а один stop-gradient не доказывает невозможность collapse.',
      r'\theta_t\leftarrow\tau\theta_t+(1-\tau)\theta_o,\qquad L=\|\widehat q_o-\operatorname{sg}(\widehat z_t)\|_2^2',
      'Шляпка означает L2-нормировку, sg — stop-gradient. При τ=.9 старый target=2 и новый online=4 дают target=2.2.',
      'Target-параметр 2, online-параметр после обновления 4, τ=.9: новый target=.9·2+.1·4=2.2.',
      ['Инициализируем две разные копии параметров; target не включаем в optimizer.',
       'Online predictor сравниваем с detached target другой view; градиент проходит только по online.',
       'В no_grad обновляем target через EMA до 2.2; это не копирование online=4 на каждом шаге.'],
      'Присваивание target=online создаёт общие веса и разрушает роль медленной target-ветви.',[
       ('Почему BYOL нельзя описать как SimCLR с пустым списком negatives?','У BYOL другая цель и асимметрия: predictor online, detached target и EMA. Простое удаление negatives из contrastive softmax не воспроизводит метод.'),
       ('Target=2, online=4, EMA momentum=.9. Новый target?', '2.2.',*N(2.2,r'0.9\cdot2+0.1\cdot4=2.2')),
       ('Почему target BYOL не следует передавать тому же optimizer, что online?','Target должен обновляться EMA, а не тем же градиентным шагом. Иначе исчезает заданный метод обновления.'),
       ('Что обычно сохраняют для downstream из BYOL?','Encoder, формирующий представление изображения. Предобучающий predictor не является обязательной головой конечной задачи.',*S(['Encoder','Только EMA-коэффициент','Матрицу всех negatives','Только prediction target каждого train-изображения']))])

    T('ssl-beit','BEiT: дискретные visual tokens',[4],
      'BEiT предсказывает дискретные visual tokens скрытых патчей, полученные от отдельного tokenizer.',
      'Входной patch embedding в выбранных позициях заменяется mask embedding. Target — индекс дискретного токена, вычисленный заранее предобученным dVAE tokenizer, а не RGB-пиксель и не текстовый token ID. ViT выдаёт logits по словарю visual codes. Cross-entropy берут по выбранным позициям. Внешний tokenizer — дополнительная часть pipeline: качество и версия токенизатора влияют на цель. Не переносите эту схему автоматически на любой masked image model: MAE и teacher-feature prediction используют иные targets.',
      r'L_{\rm mask}=-\frac1{|M|}\sum_{i\in M}\log p_\theta(v_i\mid x_{\rm masked})',
      'M — позиции маски, v_i — дискретный visual code. Число logits на позицию равно размеру visual vocabulary, а не числу RGB-каналов.',
      '4 патча, маскируем позиции 1 и 3; словарь из 8 codes. Для loss нужны 2 target IDs и logits формы [2,8].',
      ['Tokenizer по исходной картинке задаёт visual targets, например [2,5,1,7].',
       'В ViT заменяем входы позиций 1 и 3 на mask embedding, но targets 5 и 7 сохраняем отдельно.',
       'Берём CE по двум скрытым позициям: модель не должна получать сами target IDs как открытый вход.'],
      'Mask embedding на входе и target visual token в loss выполняют разные роли.',[
       ('Что является target BEiT: пиксели или code ID?', 'Дискретный visual code из предобученного tokenizer. Не непосредственная RGB-реконструкция.'),
       ('Две masked позиции и visual vocabulary=8. Сколько logits участвует в их CE?', '2·8=16 чисел.',*N(16,r'2\cdot8=16')),
       ('Зачем BEiT внешний tokenizer?', 'Он определяет дискретное пространство targets; ViT учится предсказывать эти коды по повреждённому входу.'),
       ('Можно ли положить правильный visual code скрытого патча во вход и назвать это masked prediction?', 'Нет: это передача ответа. Target нужен loss, а скрытая входная позиция должна оставаться замаскированной.')])

    T('ssl-ibot','iBOT: online tokenizer и patch targets',[5],
      'iBOT использует EMA teacher как online tokenizer для self-distillation глобальных и patch-представлений.',
      'В отличие от BEiT не требуется отдельно фиксировать dVAE tokenizer. Teacher формирует распределения targets, student предсказывает их по masked-входу. Сочетаются image-level CLS objective и patch objective. Нельзя сопоставлять patch номер 7 разных произвольных crops только по одинаковому индексу: смысл patch loss требует согласования соответствующих пространственных областей. Teacher targets отделены от графа, параметры teacher обновляются EMA. Это объясняет происхождение targets, но не делает target обычной ручной class label.',
      r'L_{\rm patch}=-\frac1{|M|}\sum_{i\in M}\sum_c p^t_{ic}\log p^s_{ic}',
      'i — согласованная masked-позиция, c — измерение распределения teacher. Teacher distribution — soft target, поэтому это CE распределений.',
      'У двух crops patch с индексом 0 может соответствовать разным углам исходной картинки. Индекс одинаков, пространственный target — нет.',
      ['Проверяем геометрию crops и выбираем действительно соответствующие позиции.',
       'Получаем detached распределение teacher и log_softmax student для каждой выбранной позиции.',
       'Усредняем patch CE по маске; отдельно учитываем image-level objective.'],
      'Одинаковый индекс токена не доказывает пространственное соответствие crops.',[
       ('В чём принципиальная замена tokenizer между BEiT и iBOT?', 'BEiT опирается на отдельный pretrained discrete tokenizer; iBOT получает targets от обучающегося EMA teacher.'),
       ('Почему patch 0 одного crop нельзя вслепую сопоставить patch 0 другого?', 'Они могут смотреть на разные области исходника. Нужна согласованная геометрия.'),
       ('Что верно о teacher target iBOT?', 'Он не получает градиент через student loss; teacher обновляют EMA.',*S(['Detached target от EMA teacher','Ручная метка породы собаки','RGB-среднее всего датасета','Target всегда равен самому student с общим Parameter'])),
       ('Teacher p=[.25,.75], student p=[.5,.5]. Чему равна CE в натуральном логарифме?', '−(.25+.75)ln(.5)=ln2≈.693147.',*N(0.69314718056,r'H(p_t,p_s)=-\log0.5=\log2'))])

    T('ssl-dino','DINOv1: multi-crop и self-distillation',[6],
      'DINO обучает student повторять распределение teacher на других views с EMA, centering и sharpening.',
      'Teacher видит global crops, student — global и local. Пара одной и той же global view исключается из cross-view сравнения. Выход teacher центрируется и преобразуется softmax с температурой; student использует собственную температуру. Center обновляется по статистикам teacher logits, EMA — по параметрам student. Это разные скользящие состояния. Centering сдерживает доминирование одного измерения, sharpening делает targets более определёнными; вместе с recipe они противодействуют разным формам collapse.',
      r'p_t=\operatorname{softmax}((u_t-c)/T_t),\qquad H(p_t,p_s)=-\sum_k p_{t,k}\log p_{s,k}',
      'c — вектор центра logits, T_t — teacher temperature. Выходные измерения этой головы не обязаны быть размеченными классами downstream-задачи.',
      'Teacher получает 2 global views, student — 2 global+4 local. Из 2×6 пар исключаем 2 совпадающие global view: остаётся 10 направленных сравнений.',
      ['Создаём две global и четыре local views; teacher применяем только к двум global.',
       'Для teacher softmax вычитаем center, student получает log_softmax; target detach.',
       'Складываем 10 cross-view CE, обновляем student optimizer, teacher EMA и center отдельно.'],
      'EMA параметров и EMA центра — не одна операция, а высокая image accuracy не гарантирует хорошую dense map.',[
       ('2 global crops teacher и 6 views student. Сколько cross-view пар после исключения одинаковых global?', '2·6−2=10.',*N(10,r'2\cdot6-2=10')),
       ('Зачем DINO разные роли centering и sharpening?', 'Centering уменьшает доминирование одного выходного измерения; sharpening уменьшает энтропию targets. Их баланс важнее лозунга «softmax не даёт collapse».'),
       ('Почему teacher DINO не обрабатывает все local crops в исходной схеме?', 'Global views дают teacher более полный контекст, student учится согласовывать с ним также локальные наблюдения.'),
       ('По чему обновляют center DINO?', 'По teacher logits/statistics, а не EMA весов student.',*S(['По статистикам teacher logits','По порядковым номерам patch','По истинным test labels','По learning rate optimizer']))])

    T('ssl-dinov2','DINOv2: image и patch objectives',[7,10,11],
      'DINOv2 соединяет image-level self-distillation с masked patch objective и масштабированием curated-данных.',
      'В лекции отдельно выделены DINO image objective, iBOT patch objective, балансировка teacher targets через Sinkhorn-Knopp, KoLeo regularization и distillation крупных моделей в меньшие. Не сводите улучшение к одной новой loss: важны данные и training recipe. CLS служит global-признаком; patch tokens сохраняют dense-информацию. Один хороший результат linear probe оценивает только конкретный протокол и не доказывает одинаковое качество всех пиксельных задач.',
      r'X_{\rm patches}\in\mathbb R^{B\times(H/P)(W/P)\times D}',
      'При размерах, кратных P, число spatial tokens равно (H/P)(W/P). CLS и возможные дополнительные tokens считают отдельно по конфигурации checkpoint.',
      'Изображение 224×224, patch size14: сетка16×16=256 patches. CLS добавляет global token, но не ячейку сетки.',
      ['Processor определяет реальный размер pixel_values; проверяем его до вычисления сетки.',
       'Из last_hidden_state отделяем CLS и patch-последовательность с учётом конфигурации модели.',
       'Получаем [1,256,D] dense features или [1,D] CLS для разных downstream-протоколов.'],
      'Не вычисляйте spatial-grid только по исходному JPEG: processor мог изменить размер или обрезать изображение.',[
       ('224×224 и P=14: сколько spatial patch tokens?', '16²=256, без CLS.',*N(256,r'(224/14)^2=256')),
       ('Почему DINOv2 — больше, чем переименование DINOv1?', 'Изменены сочетание image/patch objectives, curated data, regularization, масштабирование и distillation; это целая recipe.'),
       ('Какой признак использует frozen DINOv2 baseline нового ДЗ?', 'CLS token последнего слоя, затем L2-нормировка для retrieval. Baseline не обучает новую голову.'),
       ('Зачем отдельно оценивать CLS и patch features?', 'CLS описывает изображение в целом, patch features — spatial-представление. Хорошая global classification не гарантирует качественные dense maps.')])

    T('ssl-dinov3','DINOv3: Gram anchoring и dense features',[7,12,13,14],
      'В DINOv3 Gram anchoring поддерживает структуру сходства patch-признаков при долгом обучении.',
      'Лекция отмечает возможное расхождение: global-метрики растут, а dense maps деградируют. Gram anchoring сравнивает отношения между patch features с опорной моделью, не требуя совпадения каждой координаты признака. Важно различать архитектуру checkpoint, число register tokens и spatial patches. Для dense similarity сначала выбирают правильный patch index, затем нормируют признаки и считают cosine. Точное совпадение training loss между версиями не следует из одной иллюстративной формулы Gram.',
      r'G=\widehat F\widehat F^\top,\qquad G_{ij}=\operatorname{cos}(F_i,F_j)',
      'F имеет N строк-патчей и D признаков; иллюстративный Gram имеет форму N×N. При ортогональном вращении feature basis отношения сохраняются. Это объяснение структуры, а не полная реализация loss DINOv3.',
      'Два нормированных патча [1,0] и [0,1] дают Gram=[[1,0],[0,1]]. Поворот обоих на одинаковый угол не меняет эту Gram-матрицу.',
      ['Выделяем spatial tokens и нормируем каждую строку F по feature-оси.',
       'Умножаем F на Fᵀ: получаем сходства патч–патч, а не матрицу канал–канал.',
       'Сравниваем структуру с anchor-представлением, отдельно оцениваем global и dense качество.'],
      'Нельзя путать Gram F·Fᵀ по патчам с Fᵀ·F по каналам: это разные матрицы.',[
       ('F имеет shape [16,8]. Какова форма patch Gram F·Fᵀ?', '[16,16], то есть 256 чисел.',*N(256,r'16\cdot16=256')),
       ('Какую проблему DINOv3 адресует Gram anchoring?', 'Деградацию dense patch features на долгих schedules, которая может быть скрыта растущими global-метриками.'),
       ('Обязаны ли координаты F совпасть, если одинаковы Gram-матрицы?', 'Нет: общий ортогональный поворот признаков сохраняет попарные скалярные произведения.'),
       ('Почему перед reshape DINOv3 tokens нужно учитывать registers?', 'Register tokens не соответствуют spatial-ячейкам. Включение их в сетку нарушает число и смысл patch positions.')])

    T('ssl-dense','Dense features: patches, PCA и координаты',[11,14],
      'Dense visualization связывает токен с областью обработанного изображения и отображает его признаки.',
      'При P×P patches координаты x,y относятся к той картинке, которую реально получил encoder. Индекс ячейки — row=floor(y/P), col=floor(x/P), плоский index=row·W_p+col. Для PCA-красок нескольких изображений в лекции используется одна совместно обученная PCA: разные PCA создают несопоставимые RGB-оси. Цвет — координата проекции, а не семантическая маска. Если вход pad/resize/crop, координаты клика надо перевести в обработанную систему.',
      r'i=\lfloor y/P\rfloor W_p+\lfloor x/P\rfloor',
      'Индексы с нуля, W_p — ширина patch-grid. Вектор z_i можно сравнить со всеми нормированными patches, построив similarity map.',
      'Кадр56×28, P=14: сетка2 строки×4 столбца. Точка x=31,y=15 попадает row1,col2,index6.',
      ['Проверяем координаты на обработанном кадре: 31<56 и 15<28.',
       'Делим на14 с округлением вниз: col2,row1; индекс1·4+2=6.',
       'Берём седьмую spatial-строку признаков; CLS/register не входят в эту нумерацию.'],
      'Цвет PCA не является номером класса; одинаковые цвета из независимо fit PCA нельзя автоматически сравнивать.',[
       ('W_p=4, P=14, x=31,y=15. Плоский индекс patch?', '6.',*N(6,r'\lfloor15/14\rfloor\cdot4+\lfloor31/14\rfloor=6')),
       ('Почему независимая PCA для каждого изображения затрудняет сравнение цветов?', 'Каждая PCA выбирает собственный базис и знаки компонент. Одинаковый RGB уже не означает одинаковое направление в исходном feature space.'),
       ('Что нужно проверить перед reshape patch tokens в H_p×W_p?', 'Реальный обработанный размер, patch size, число spatial tokens и исключение CLS/register tokens.'),
       ('Доказывают ли похожие цвета PCA точную сегментацию объекта?', 'Нет. Это низкоразмерная визуализация features, не обученный classifier пикселей и не GT-mask.')])

    T('ssl-probe','Frozen features, k-NN и linear probe',[15,16],
      'Linear probe обучает только линейный classifier поверх фиксированных признаков backbone.',
      'Backbone переводят в eval, его параметры выключают из градиента, признаки можно один раз закешировать. Classifier D→C обучается с labels train. Fine-tuning обновляет также backbone; frozen k-NN вообще не обучает classifier, а голосует по соседям. Эти протоколы отвечают на разные вопросы. Если аугментации случайны, кеш один раз фиксирует конкретные views: это уже часть выбранного протокола. Настройки probe подбирают на validation, итоговый test не используют для scheduler/checkpoint selection.',
      r'\ell(x)=Wf_{\rm frozen}(x)+b,\qquad \#\theta_{\rm head}=DC+C',
      'D — размер embedding, C — число классов. Forward backbone не должен строить train-граф, но head всё ещё обучается.',
      'D=8,C=3: linear head содержит 8·3+3=27 параметров. Обучение этих 27 не меняет исходные веса backbone.',
      ['Один раз извлекаем features [N,8] в eval/no_grad и сохраняем соответствующие labels.',
       'Head Linear(8,3) выдаёт logits [N,3], CE обновляет только 27 его параметров.',
       'Проверяем frozen k-NN отдельно: это другой baseline, без обучения этой головы.'],
      'requires_grad=False не выключает dropout и не замораживает BatchNorm running statistics без eval.',[
       ('D=8,C=3, head с bias. Сколько обучаемых параметров linear probe?', '27.',*N(27,r'8\cdot3+3=27')),
       ('В чём различие linear probe и fine-tuning?', 'В probe backbone фиксирован и обучается линейный classifier; fine-tuning обновляет хотя бы часть backbone.'),
       ('Зачем frozen backbone переводить ещё и в eval?', 'Чтобы dropout и обновление BatchNorm running statistics не меняли определение извлекаемых features.'),
       ('Можно ли сравнивать raw frozen DINO retrieval и linear probe как один и тот же протокол?', 'Нет. В raw retrieval нет обученной supervised-головы, linear probe использует labels для обучения classifier.')])

    T('metric-calibration','SigLIP: calibration и границы open-set rejection',[21,22],
      'Независимые image–text scores SigLIP не являются автоматически вероятностями взаимоисключающих классов.',
      'Closed-set baseline выбирает argmax logits. One-vs-rest Platt calibration подгоняет sigmoid(a_k·l_k+b_k) на отдельном calibration split. Threshold для отказа выбирают на validation, затем фиксируют для test. Отдельные OVR scores не обязаны суммироваться в1. Более высокая калибровка сама по себе не гарантирует хорошее ранжирование или распознавание неизвестных классов. В показанном ноутбуке все test-примеры относятся к выбранным known classes: эксперимент измеряет rejection на known data, а полноценный open-set тест должен включать unknown identities/classes.',
      r'\widehat p_k=\sigma(a_k\ell_k+b_k),\quad \widehat y=\begin{cases}\arg\max_k\widehat p_k,&\max_k\widehat p_k\ge t\\-1,&\text{иначе}\end{cases}',
      'Параметры a,b обучаются на calibration, t выбирается на validation. Число −1 — соглашение для rejected, не новый обученный semantic-класс.',
      'Scores [.4,.35,.2], threshold=.5 → отказ −1. Без threshold closed-set всё равно выбрал бы класс0.',
      ['Разделяем calibration, validation и test до подбора параметров.',
       'На validation фиксируем t=.5; на query берём maximum=.4 и отказываемся от ответа.',
       'Чтобы оценить unknown rejection, добавляем независимые неизвестные классы в test-протокол.'],
      'Reject rate на known-only test не является доказательством хорошего open-set распознавания.',[
       ('Scores [.4,.35,.2], t=.5: какой ответ rejection-правила?', '−1, ни один score не достиг порога.',*N(-1,r'\max(0.4,0.35,0.2)<0.5\Rightarrow\widehat y=-1',0)),
       ('Почему OVR sigmoid scores не обязаны суммироваться в1?', 'Каждый бинарный calibrator обучен отдельно. Это не общий softmax с единой нормировкой.'),
       ('Можно ли подбирать threshold на test и затем считать его независимой оценкой?', 'Нет. Подбор использует информацию test; threshold фиксируют на validation.'),
       ('Чего не хватает known-only test для полноценной оценки open-set?', 'Примеров unknown classes и заранее заданного протокола оценки их ложного принятия/отказов.')])

    T('metric-tasks','Verification, identification и ReID',[23,24],
      'Verification сравнивает пару, identification ищет identity среди gallery, ReID сопоставляет объект между наблюдениями.',
      'Verification — решение same/different по паре и threshold score. Identification — поиск среди многих identities, closed-set или с отказом. ReID часто требует retrieval при смене камеры, позы и освещения. Один encoder может использоваться во всех трёх сценариях, но labels, sampling и метрики отличаются. Нельзя автоматически переносить высокий classifier accuracy на verification при очень низком FAR.',
      r's(x_1,x_2)=\frac{f(x_1)^\top f(x_2)}{\|f(x_1)\|\|f(x_2)\|}',
      'Для verification score сравнивают с threshold; для identification ранжируют gallery. Нулевой вектор требует отдельной обработки.',
      'Пара паспорт–селфи: один score и порог. Запрос к gallery из1000 identities: тысяча кандидатов и порядок соседей.',
      ['Определяем, вопрос о двух объектах или о выборе identity из gallery.',
       'Для пары используем threshold, для gallery — ранжирование и top-K.',
       'Считаем метрику нужного сценария: TAR@FAR или retrieval/CMC вместо произвольной accuracy.'],
      'Один и тот же embedding не делает одинаковыми протоколы verification и identification.',[
       ('Чем verification отличается от identification?', 'Verification решает, одна ли identity у пары; identification ищет identity среди множества gallery-кандидатов.'),
       ('Задача «найти того же пешехода на другой камере» относится к чему?', 'ReID с retrieval/identity matching между наблюдениями.',*S(['ReID','Только semantic segmentation','Только next-token prediction','Только pixel MSE'])),
       ('Почему top-1 classifier accuracy недостаточно для оценки verification при низком FAR?', 'Иной набор пар и операционная точка. Нужны распределения genuine/impostor scores и TAR при требуемом FAR.'),
       ('Почему gallery должна быть описана вместе с метрикой identification?', 'Размер, состав и количество релевантных кандидатов меняют сложность поиска и смысл top-K/CMC.')])

    T('retrieval-recall','Semantic Recall@K, Hit@K и Precision@K',[24],
      'Recall@K — доля найденных релевантных объектов; Hit@K — наличие хотя бы одного; Precision@K — их доля в выдаче.',
      'В этом курсе semantic relevance задаётся разметкой. Для query q множество R_q состоит из всех релевантных gallery-объектов. Recall делит найденные positives на |R_q|, precision — на K, hit — бинарный индикатор. В metric-learning литературе названием Recall@K иногда обозначают Hit@K, поэтому сначала проверяют формулу. В ДЗ7 эти названия разделены явно. Если релевантных gallery-объектов нет, нужен заранее оговорённый способ учёта такого query.',
      r'R@K=\frac{|R_q\cap topK|}{|R_q|},\quad P@K=\frac{|R_q\cap topK|}{K},\quad H@K=\mathbf1[|R_q\cap topK|>0]',
      'Для одного query с четырьмя positives в gallery и одним найденным в top2 получаем recall1/4, precision1/2, hit1.',
      'R_q содержит4 объекта; top2 содержит1 из них: Recall@2=.25, Precision@2=.5, Hit@2=1.',
      ['Считаем число релевантных во всей gallery: R=4.',
       'В первых двух результатах находим один relevant: делим на4 для recall и на2 для precision.',
       'Поскольку один relevant найден, hit=1; он не сообщает, что три positives потеряны.'],
      'Не называйте Hit@K semantic Recall@K без явного соглашения.',[
       ('В gallery24 positives, первый сосед правильный. Semantic Recall@1 по ДЗ7?', '1/24≈.041667, а Hit@1=1.',*N(1/24,r'R@1=1/24')),
       ('R=4, в top2 найден1 relevant. Precision@2?', '.5.',*N(.5,r'P@2=1/2')),
       ('Когда Hit@K и semantic Recall@K совпадают?', 'Если для каждого query ровно один релевантный gallery-объект; также в частных крайних случаях, но не в общем.'),
       ('Почему semantic Recall@1 в ДЗ7 не может достигнуть1 при24 positives?', 'В одном месте выдачи помещается только один из24 релевантных объектов, поэтому верхняя граница1/24.')])

    T('retrieval-map','AP и mAP: полное ранжирование',[24],
      'Average Precision усредняет precision на позициях релевантных результатов; mAP усредняет AP по queries.',
      'В полном AP знаменатель равен общему числу релевантных gallery-объектов, а ранжирование охватывает всю gallery. mAP retrieval не использует детекторские IoU thresholds: это другое определение релевантности, хотя аббревиатура та же. Для AP@K существуют разные соглашения о знаменателе, поэтому усечённую метрику подписывают явно. В новом ДЗ для mAP требуется вся gallery, не top10.',
      r'AP(q)=\frac1{|R_q|}\sum_{r=1}^{G}P@r\cdot rel_q(r),\quad mAP=\frac1Q\sum_q AP(q)',
      'Для relevant на рангах1 и3: P@1=1, P@3=2/3, AP=(1+2/3)/2=5/6.',
      'Ранжирование [relevant, irrelevant, relevant], всего2 positives: AP=5/6≈.8333.',
      ['Строим индикатор relevant=[1,0,1] по labels query и ранжированной gallery.',
       'Precision по рангам равен [1,1/2,2/3]; в сумму входят только ранги1 и3.',
       'Делим 1+2/3 на2; затем для mAP повторяем по всем query и усредняем.'],
      'Top10 недостаточно для полного AP, если в gallery релевантные объекты встречаются ниже.',[
       ('Relevant находятся на рангах1 и3, всего2. Полный AP?', '5/6.',*N(5/6,r'AP=(1+2/3)/2=5/6')),
       ('AP двух queries .5 и1. Чему равен mAP?', '.75.',*N(.75,r'mAP=(0.5+1)/2=0.75')),
       ('Зачем ДЗ7 запрашивает всю gallery для mAP?', 'Для суммы precision на всех relevant ranks и корректного общего знаменателя, а не усечённой top-K оценки.'),
       ('Можно ли к retrieval mAP автоматически приписать @[.5:.95]?', 'Нет. Это IoU-протокол детекции. Для retrieval relevance определяется объектами/метками, а не обязательно overlap рамок.')])

    T('retrieval-ndcg','NDCG и CMC: важность порядка',[24],
      'NDCG учитывает градуированную полезность и ранг; CMC показывает, найден ли первый верный identity до заданного ранга.',
      'В DCG gain=(2^rel−1), discount=log2(rank+1). IDCG строится из идеальной перестановки тех же кандидатов, после чего DCG нормируют. Если IDCG=0, нужен явный договор, обычно возвращают0. CMC/rank-K ориентирован на первое правильное совпадение и не описывает полноту всех positives. Для нескольких нужных объектов mAP и recall дают дополнительную информацию.',
      r'DCG@K=\sum_{i=1}^K\frac{2^{rel_i}-1}{\log_2(i+1)},\quad NDCG=DCG/IDCG',
      'При relevance [0,1] DCG@2=1/log2(3); идеал[1,0] даёт1, значит NDCG≈.63093.',
      'Два кандидата с relevance0 и1 пришли в плохом порядке: NDCG@2≈.63093, хотя Hit@2=1.',
      ['Назначаем gains0 и1; второй ранг дисконтируется log2(3).',
       'Идеальная выдача ставит relevant первым, IDCG=1.',
       'Нормируем; Hit@2 и CMC@2 могут быть1, но NDCG отражает потерю порядка.'],
      'У IDCG должен быть тот же набор кандидатов и K, а не произвольно удобная идеальная выдача.',[
       ('Relevance [0,1], K=2. NDCG?', '1/log2(3)≈.63093.',*N(.63092975357,r'NDCG=\frac{1/\log_2 3}{1}')),
       ('Что означает CMC@5=.8?', 'У80% запросов первый правильный identity найден не ниже пятого ранга при данном протоколе.'),
       ('Почему CMC не заменяет recall для большого множества positives?', 'CMC достаточно первого правильного match; остальные релевантные могут быть пропущены.'),
       ('Что делать с IDCG=0?', 'Заранее определить соглашение, например NDCG=0 или исключение query; нельзя молча делить на0.')])

    T('metric-verification','ROC-AUC и TAR при заданном FAR',[24],
      'TAR — доля принятых genuine-пар, FAR — доля ошибочно принятых impostor-пар при выбранном пороге.',
      'Для cosine similarity принимаем пару, если score≥threshold; повышение threshold обычно снижает обе доли. ROC-AUC агрегирует весь диапазон порогов и может скрывать плохое качество в нужной области очень малого FAR. Порог фиксируют на validation, не на итоговом test. Оценка редких ошибок требует достаточно много impostor-пар: ноль ошибок на маленьком наборе не доказывает нулевой истинный FAR.',
      r'TAR=\frac{TP}{TP+FN},\qquad FAR=\frac{FP}{FP+TN}',
      'Положительные пары — same identity. У1000 impostor-пар один false accept даёт FAR=.001.',
      '100 genuine:90 приняты. 1000 impostor:1 принят. TAR=.9, FAR=.001.',
      ['Разделяем genuine и impostor, не смешивая их знаменатели.',
       'Применяем один зафиксированный threshold и получаем90 TP и1 FP.',
       'Делим на100 и1000 соответственно; для низкого FAR проверяем достаточность числа пар.'],
      'Высокий ROC-AUC не гарантирует высокий TAR именно при FAR=10⁻⁵.',[
       ('1 false accept на1000 impostor pairs. FAR?', '.001.',*N(.001,r'FAR=1/1000=0.001')),
       ('90 принятых genuine из100. TAR?', '.9.',*N(.9,r'TAR=90/100=0.9')),
       ('Почему ROC-AUC недостаточно для банковского verification с заданным очень низким FAR?', 'Нужна конкретная операционная точка TAR@FAR; площадь по всем порогам не описывает её отдельно.'),
       ('Можно ли по0 ошибкам на100 impostor pairs доказать FAR≤10⁻⁵?', 'Нет. Выборка слишком мала для уверенного вывода о столь редком событии.')])

    T('metric-triplet','Triplet loss: margin и расстояние',[25,26],
      'Triplet loss требует, чтобы anchor был ближе к positive, чем к negative, с заданным запасом margin.',
      'L=max(d(a,p)−d(a,n)+m,0). Уточняйте, используется обычное евклидово расстояние или его квадрат: значение loss и подходящий margin различаются. Hard negative ближе positive, semi-hard находится дальше positive, но нарушает margin, easy уже удовлетворяет ограничению. Zero loss одной тройки не доказывает хорошее embedding space на всём датасете.',
      r'L=[d(a,p)-d(a,n)+m]_+,\quad \text{semi-hard: }d(a,p)<d(a,n)<d(a,p)+m',
      'В ДЗ7 используется обычный Euclidean d без квадрата. При d_ap=.4,d_an=.5,m=.2 loss=.1.',
      'Anchor0, positive.4, negative.5 на прямой; margin.2. Loss=max(.4−.5+.2,0)=.1.',
      ['Вычисляем обе дистанции в одной выбранной метрике, не смешивая d и d².',
       'Negative дальше positive, но на.1 вместо нужных.2 — это semi-hard.',
       'Loss=.1 показывает недостающий запас; если negative отнести на.8, loss станет0.'],
      'Margin измеряется в единицах выбранного расстояния; квадрат расстояния меняет задачу.',[
       ('d_ap=.4,d_an=.5,m=.2. Triplet loss?', '.1.',*N(.1,r'[0.4-0.5+0.2]_+=0.1')),
       ('Как называется negative, если d_ap<d_an<d_ap+m?', 'Semi-hard: дальше positive, но нарушает запас.',*S(['Semi-hard','Easy с нулевым loss','Self-positive','Ground-truth center'])),
       ('Почему переход d→d² при том же margin не сохраняет loss?', 'Нелинейное преобразование меняет разности расстояний и шкалу margin, хотя порядок соседей может сохраниться.'),
       ('Почему loss=0 на выбранных easy triples не доказывает хорошее обучение?', 'Sampler/miner мог не показывать модели трудные случаи; нулевой loss касается только этих троек.')])

    T('metric-mining','Online batch-hard mining: маски и экстремумы',[27],
      'Batch-hard выбирает для каждого anchor самого далёкого positive и ближайшего negative внутри текущего батча.',
      'Строят B×B distance matrix, same-label mask и отдельную диагональную mask. Anchor не может быть своим positive. Для max-positive запрещённые позиции заполняют −∞, для min-negative — +∞. Проверяют, что у каждого anchor есть хотя бы один positive и negative. Sampler определяет доступный pool; miner переоценивает сложность по текущим embeddings на каждом шаге. Самые трудные negatives могут оказаться ошибками разметки.',
      r'L=\frac1B\sum_a[\max_{p\ne a:y_p=y_a}d_{ap}-\min_{n:y_n\ne y_a}d_{an}+m]_+',
      'Стоимость хранения полной матрицы O(B²). Использование0 вместо∞ для запрещённых negatives искусственно создаёт идеального nearest negative.',
      'От anchor positive distances [.2,.7], negative [.6,1.1], m=.2: hard positive=.7, negative=.6, loss=.3.',
      ['По labels выделяем два positive и два negative, диагональ исключаем.',
       'Среди positive берём max=.7, среди negatives min=.6.',
       'Получаем .7−.6+.2=.3; после обновления embeddings выбор может измениться.'],
      'Если в батче один объект каждого класса, маска positive пуста: это ошибка протокола, а не повод усреднить −∞.',[
       ('Hard positive=.7, hard negative=.6, margin=.2. Loss anchor?', '.3.',*N(.3,r'[0.7-0.6+0.2]_+=0.3')),
       ('Почему запрещённые negatives нельзя заполнить нулями перед min?', 'Ноль выиграет минимум и станет фиктивным самым близким negative. Нужен +∞.'),
       ('Почему anchor исключают из positive mask?', 'Расстояние до себя0 не является независимым positive и может исказить выбор/валидность.'),
       ('В чём online mining отличается от заранее сохранённых hardest triples?', 'Online заново выбирает пары по текущим embeddings внутри batch; сохранённые triples устаревают при обучении.')])

    T('metric-pk','P×K sampler и состав batch',[9,10,13],
      'P×K sampler выбирает P разных классов и K примеров каждого для формирования metric-learning batch.',
      'Для batch-hard нужны P≥2 и K≥2: у anchor будет K−1 других positives и (P−1)K negatives. Sampling и mining решают разные задачи. Если у класса меньше K доступных изображений, заранее определяют выбор с replacement или ограничения классов. Повтор одного и того же изображения не равнозначен независимому positive. Балансировка batch по классу меняет training distribution и не должна незаметно менять протокол test.',
      r'B=PK,\quad n_+=K-1,\quad n_-=(P-1)K',
      'Для P=6,K=4 batch содержит24 объекта, каждому anchor доступны3 positives и20 negatives.',
      'P=6,K=4 → B=24. Другая картинка той же identity может быть positive, но сам anchor исключается.',
      ['Выбираем6 разных классов и по4 различных изображения каждого, если данных достаточно.',
       'Для каждого из24 anchors строим same-class mask без диагонали:3 positives.',
       'Остальные5×4=20 объектов — candidates negative; miner выбирает трудные из них.'],
      'Обычный random shuffle не гарантирует хотя бы два примера одной identity в каждом batch.',[
       ('P=6,K=4. Сколько negatives доступно одному anchor?', '20.',*N(20,r'(6-1)\cdot4=20')),
       ('P=6,K=4. Сколько других positives у anchor?', '3.',*N(3,r'4-1=3')),
       ('Почему случайный batch из24 объектов не заменяет гарантированный P×K?', 'Он может содержать по одному объекту разных identities, оставив пустые positive masks.'),
       ('Чем sampler отличается от miner?', 'Sampler задаёт состав batch; miner выбирает пары/тройки внутри доступного состава по embeddings.')],path=HOMEWORK)

    T('face-normalized','NormSoftmax: нормированные веса и scale',[28,29,32],
      'NormSoftmax классифицирует по косинусам между нормированным embedding и нормированными центрами классов.',
      'Нормируют как features, так и строки classifier weight. Тогда logits=s·cosθ лежат в [−s,s]. Scale s задаёт концентрацию softmax и масштаб градиентов; нормировка сама не заменяет этот параметр. В обучении используют class centers и labels, а на inference для verification/retrieval сравнивают embeddings. Двумерные точки лекции — иллюстрация, реальная feature dimension обычно иная.',
      r'\ell_j=s\,\frac{z^\top w_j}{\|z\|\|w_j\|}',
      'При z=[3,4] нормированное z=[.6,.8]. Для центров[1,0] и[0,1] и s=10 logits=[6,8].',
      'z=[3,4], центры осей, scale10: logits6 и8; победил второй класс, но training-class centers не обязательны в retrieval.',
      ['Норма z равна5, поэтому feature после нормировки [.6,.8].',
       'Веса центров уже нормированы; скалярные произведения .6 и.8.',
       'Умножаем на scale10 и передаём logits[6,8] в CE без ручного softmax.'],
      'Нормировка только features при ненормированных weights оставляет зависимость от нормы центров.',[
       ('Cosines .6 и.8, scale10. Logit второго класса?', '8.',*N(8,r'10\cdot0.8=8')),
       ('Зачем scale s после нормировки?', 'Он меняет диапазон logits, концентрацию softmax и градиенты; ограниченные cosines без scale могут дать слишком мягкое распределение.'),
       ('Нужны ли обученные class centers для verification новых identities?', 'Обычно сравнивают нормированные embeddings без training classifier centers.'),
       ('Достаточно ли нормировать только feature для чистых cosine logits?', 'Нет. Нужно нормировать и class weights; иначе их нормы масштабируют logits по-разному.')])

    T('face-margins','SphereFace, CosFace и ArcFace',[28,29,32],
      'Face margins усложняют правильный target-logit при обучении, чтобы повысить разделимость identities на сфере.',
      'CosFace вычитает margin из cosθ_y, ArcFace добавляет margin к θ_y, SphereFace умножает угол. Эти операции не эквивалентны: угол измеряют в радианах, cosine — безразмерный. Меняют target-logit, остальные классы оставляют базовыми. Для SphereFace нужна кусочная монотонная реализация; и у ArcFace нужно следить за диапазоном углов/численной устойчивостью. На inference labels неизвестны, target-margin не применяют.',
      r'\text{CosFace: }s(\cos\theta_y-m),\quad\text{ArcFace: }s\cos(\theta_y+m),\quad\text{SphereFace: }s\cos(m\theta_y)',
      'Например θ=π/3, Arc margin=π/6 даётcos(π/2)=0; CosFace с m=.1 даёт.5−.1=.4. Значения m имеют разный смысл.',
      'θ_y=60°, ArcFace margin30°: target cosine меняется с.5 на0. Другой класс с cosine.2 не получает этот margin.',
      ['Из нормированных embeddings/weights получаем cosine всех классов.',
       'Для известного y меняем target: cos(60°+30°)=0; остальные logits сохраняем.',
       'Умножаем на scale и считаем CE; на inference используем embeddings без training labels.'],
      's·cos(θ+m) не равно s·(cosθ−m); один и тот же численный margin нельзя механически переносить между loss.',[
       ('CosFace: cosine_y=.8, margin=.1, scale10. Target-logit?', '7.',*N(7,r'10(0.8-0.1)=7')),
       ('ArcFace: θ=π/3,m=π/6. Target cosine?', '0, потому что сумма угловπ/2.',*N(0,r'\cos(\pi/3+\pi/6)=\cos(\pi/2)=0')),
       ('Почему target margin не применяют к неизвестной identity на inference?', 'Target label отсутствует; margin — обучающее условие для classifier, а inference обычно сравнивает embeddings.'),
       ('Почему SphereFace нельзя безоговорочно реализовать только cos(mθ) на всём диапазоне?', 'При m>1 косинус перестаёт быть монотонным по исходному углу; требуется кусочная монотонная конструкция.')])

    T('face-adaface','AdaFace: качество и два компонента margin',[29],
      'AdaFace использует норму feature до L2-нормировки как proxy качества и адаптирует angular и additive margin.',
      'Сначала сохраняют исходную норму z, отделяют её от графа, стандартизуют running mean/std и ограничивают диапазон. После L2-нормировки длина вектора1 уже не несёт эту информацию. Нормированный proxy качества влияет на две составляющие margin, поэтому AdaFace нельзя заменить одним cos(θ+m(norm)). Proxy статистический: высокая норма не является гарантированным измерением физического качества каждой картинки. EMA статистики — часть состояния обучения.',
      r'q=\operatorname{clip}\!\left(h\frac{\operatorname{sg}(\|z\|)-\mu}{\sigma},-1,1\right),\quad g_a=-mq,\quad g_b=mq+m,\quad f_y=s(\cos(\theta_y+g_a)-g_b)',
      'Здесь q — detached quality proxy, h — масштаб стандартизации, σ>0. При q=0 angular shift0, additive margin=m; это центральный случай, не весь метод.',
      'q=0,m=.4: g_a=0,g_b=.4. При q=1: g_a=−.4,g_b=.8; меняются обе компоненты.',
      ['Снимаем норму до нормировки, не теряя её при превращении z в единичный вектор.',
       'Вычисляем detached стандартизованный q и clip в[−1,1].',
       'Строим обе компоненты margin; вычисление только angular term пропустило бы часть метода.'],
      'Нельзя брать норму уже L2-нормированного feature как информативный quality proxy: у ненулевых векторов она1.',[
       ('q=0,m=.4. Чему равен additive компонент g_b?', '.4.',*N(.4,r'g_b=0.4\cdot0+0.4=0.4')),
       ('Зачем AdaFace норма feature до L2-normalize?', 'После нормировки все ненулевые embeddings имеют норму1, поэтому исходный proxy качества теряется.'),
       ('Почему quality proxy отделяют от графа?', 'Чтобы механизм оценки качества не предоставлял модели прямой градиентный путь изменения нормы ради более удобного margin.'),
       ('AdaFace — это только cos(θ+m(q))?', 'Нет: адаптируются angular и additive составляющие, плюс используются running statistics и clipping.')])

    T('search-flat','Exact cosine search и FAISS Flat',[34,35],
      'Exact flat search сравнивает query со всей gallery и задаёт baseline для оценки ANN-приближения.',
      'Для cosine нормируют query и gallery по строкам, затем ищут максимум inner product через IndexFlatIP. Для L2 FAISS возвращает квадрат евклидова расстояния: порядок тот же, численные значения другие. Flat не требует обучать codebook, но хранит всю матрицу features. Нулевой вектор не имеет определённого cosine; clamp norm предотвращает NaN, но не делает его информативным. Для top-k держите 1≤k≤число gallery.',
      r'\|\widehat x-\widehat y\|_2^2=2-2\widehat x^\top\widehat y,\qquad V=4GD\ \text{байт для float32}',
      'G — число gallery-векторов, D — размерность. Это память только весов/векторов, без ID и служебных структур.',
      'Query[1,0], gallery[1,0],[0,1]: IP scores1,0; squared L2 distances0,2. Максимум IP и минимум L2 выбирают один ID.',
      ['Нормируем обе стороны, приводим матрицы к contiguous float32.',
       'IndexFlatIP добавляет gallery-строки; search возвращает scores и ID строк.',
       'Сверяем ranking с NumPy Q·Gᵀ и учитываем, что FAISS L2 выдаёт d².'],
      'В тексте ДЗ встречается FlatIPIndex, но правильный Python-класс — faiss.IndexFlatIP.',[
       ('Для единичных векторов cosine=.75. Чему равен squared L2?', '.5.',*N(.5,r'd^2=2-2\cdot0.75=0.5')),
       ('Нормировать только query достаточно для cosine-ranking по ненормированной gallery?', 'Нет. Нормы gallery влияют на inner products и могут менять порядок.'),
       ('Как называется exact inner-product индекс FAISS?', 'IndexFlatIP.',*S(['IndexFlatIP','FlatIPIndex','IndexCosineOnly','ExactCosineModel'])),
       ('1000 векторов D=128 float32. Сколько байт для самих vectors?', '512000 байт.',*N(512000,r'1000\cdot128\cdot4=512000',0))])

    T('search-hnsw','HNSW: граф и ANN recall',[34,35],
      'HNSW ускоряет поиск через многоуровневый граф, постепенно уточняя кандидатов.',
      'M управляет связностью графа, efConstruction — шириной поиска при построении, efSearch — усилием на query. Больший efSearch обычно повышает recall ценой latency, но это измеряемый компромисс, а не гарантия для любой конфигурации. На тех же embeddings строят exact top-K и сравнивают множества ID. ANN recall не использует labels и не сообщает, хорошо ли encoder отделяет semantic-классы. Изменение metric или нормировки между exact и approximate делает сравнение некорректным.',
      r'ANN\ recall@K=\frac1Q\sum_q\frac{|I_q^{\rm exact}\cap I_q^{\rm ann}|}{K}',
      'I — ID соседей в той же gallery и метрике. При ties на границе exact top-K заранее определяют правило сравнения.',
      'Exact top3=[1,2,3], HNSW top3=[1,3,8] → ANN recall=2/3. Все три могут быть неправильного semantic-класса.',
      ['Фиксируем gallery, нормировку и metric, строим exact baseline.',
       'Меняем efSearch и измеряем время/совпадение ID без изменения embeddings.',
       'Отдельно считаем semantic Recall/mAP по labels, чтобы оценить уже encoder.'],
      'ANN recall=1 может означать точное воспроизведение плохого semantic-ranking.',[
       ('Exact IDs[1,2,3], ANN IDs[1,3,8]. ANN recall@3?', '2/3.',*N(2/3,r'|\{1,2,3\}\cap\{1,3,8\}|/3=2/3')),
       ('Какой параметр HNSW меняет query-time усилие поиска?', 'efSearch.',*S(['efSearch','Число RGB-каналов','Margin ArcFace','Номер эпохи encoder'])),
       ('Нужны ли class labels для ANN recall?', 'Нет. Сравниваются ID approximate и exact neighbors; labels нужны semantic-метрикам.'),
       ('Почему высокая ANN recall не гарантирует высокий retrieval mAP?', 'Approximation может хорошо воспроизводить exact ranking плохого embedding space.')])

    T('search-ivfpq','IVF-PQ: coarse lists и сжатие',[34,35],
      'IVF выбирает часть кластерных списков, PQ сжимает вектор набором кодов подпространств.',
      'nlist — число coarse clusters, nprobe — число просматриваемых списков на запрос. Product quantization делит D на M подпространств и сохраняет по b бит на subvector. Codebook надо обучить на репрезентативных данных до add. Ошибки возможны из-за непросмотренных списков и из-за квантования расстояний. Даже nprobe=nlist не превращает PQ-коды в исходные float-векторы. Exact reranking помогает только внутри выбранного пула и требует исходных векторов.',
      r'\mathrm{bytes/code}=\left\lceil\frac{Mb}{8}\right\rceil,\qquad D\bmod M=0',
      'Для типичной равномерной PQ-разбивки D должен делиться на M. Формула не включает ID, coarse centroids, codebooks и прочие накладные расходы.',
      'D=128,M=16,b=8: код16 байт вместо512 байт float32. Это32-кратное сжатие только vector payload.',
      ['Обучаем coarse и PQ codebooks на подходящем train-пуле vectors.',
       'На query выбираем nprobe списков и ранжируем по приближённым distances PQ.',
       'Измеряем ANN recall и при необходимости rerank кандидатов исходными features.'],
      'Увеличение nprobe уменьшает ошибку отбора списков, но не устраняет quantization error само по себе.',[
       ('M=16,b=8. Сколько байт в PQ code без ID?', '16.',*N(16,r'16\cdot8/8=16',0)),
       ('Почему nprobe=nlist не гарантирует exact ranking для IVF-PQ?', 'Все списки просмотрены, но distances по PQ остаются квантованными; это другая ошибка.'),
       ('Почему IVF-PQ требует train до add?', 'Нужно выучить coarse centroids и codebooks, чтобы закодировать vectors и направить их в списки.'),
       ('Всегда ли exact rerank вернёт глобально лучшего соседа?', 'Нет. Он переупорядочит только кандидатов; потерянного на первой стадии соседа восстановить не сможет.')])

    T('search-lsh','LSH, random hyperplanes и KD-tree',[34,35],
      'LSH строит хеши, у которых близкие объекты чаще сталкиваются; выбор family зависит от метрики.',
      'Для random-hyperplane cosine LSH бит задаётся знаком rᵀx. Совпадение по одной гиперплоскости вероятнее при малом угле. Конкатенация bits и несколько tables/probes меняют компромисс recall, времени и памяти. MinHash предназначен для Jaccard sets и не является заменой cosine-hash dense-признаков. KD-tree эффективен в малой размерности, но в высокой pruning часто деградирует. Ни hash collision, ни общая leaf не доказывают semantic-сходство.',
      r'h_r(x)=\mathbf1[r^\top x\ge0],\qquad P(h_r(x)=h_r(y))=1-\theta/\pi',
      'Для ненулевых vectors и случайного направления гиперплоскости при углеπ/2 вероятность совпадения одного бита1/2.',
      'Ортогональные vectors имеют угол90°: один случайный hyperplane-bit совпадает с вероятностью.5.',
      ['Выбираем hash family под cosine, а не под несвязанную Jaccard-метрику.',
       'Считаем signs dot products и получаем bit signature.',
       'Кандидатов из buckets проверяем исходной метрикой и оцениваем recall относительно exact.'],
      'Параметры LSH не выбирают без измерения recall/latency на своей коллекции.',[
       ('Уголπ/2. Вероятность совпадения одного random-hyperplane bit?', '.5.',*N(.5,r'1-(\pi/2)/\pi=0.5')),
       ('Для какой похожести предназначен MinHash?', 'Jaccard для множеств/подходящих бинарных признаков, не произвольный cosine dense embeddings.',*S(['Jaccard sets','Любая distance без настройки','Только RGB MSE','IoU любых bounding boxes без кодирования'])),
       ('Почему KD-tree может потерять преимущество при большой D?', 'Разделения слабо отсекают кандидатов, приходится просматривать много ветвей; выигрыш над exhaustive search уменьшается.'),
       ('Одинаковый LSH bucket означает одинаковую identity?', 'Нет. Collision вероятностный и относится к метрике feature space, не гарантирует semantic-label.')])

    T('metric-protocol','ДЗ7: class-disjoint split и gallery/query',[1,4,6,7,17],
      'В новом ДЗ train, validation и test разделены по классам, а gallery и query — по изображениям внутри отложенных классов.',
      'Выбраны24 класса по40 картинок:12 train-классов дают480 train-images;6 validation и6 test классов имеют по24 gallery и16 query на класс. Отложенные identities не участвуют в supervised training CNN. Gallery используется как база поиска, query — как запросы. Это не тот же протокол, что случайный split изображений с одинаковыми train/test labels. Checkpoint и hyperparameters выбирают по validation; test оценивают после фиксации выбора. Даже frozen backbone не оправдывает подбор по test.',
      r'|D_{\rm train}|=12\cdot40,\quad |G_{val}|=6\cdot24,\quad |Q_{val}|=6\cdot16',
      'Получаем480 train,144 validation-gallery,96 validation-query; test имеет те же размеры. Всего960 изображений без пересечения image IDs.',
      'Validation содержит240 картинок:144 gallery и96 queries. У каждого query24 semantic positives в gallery.',
      ['Делим24 class IDs на12/6/6 без пересечения классов.',
       'В каждом отложенном классе выбираем24 gallery и16 разных query.',
       'Сохраняем indices/revision и проверяем выбор checkpoint на validation, а не по test mAP.'],
      'Если query-картинка попадает в gallery, self-match искусственно повышает качество.',[
       ('6 test-классов,24 gallery на класс. Размер gallery?', '144.',*N(144,r'6\cdot24=144',0)),
       ('6 test-классов,16 query на класс. Число queries?', '96.',*N(96,r'6\cdot16=96',0)),
       ('Почему случайный image split не заменяет class-disjoint split ДЗ7?', 'При image split train может видеть те же identities; class-disjoint оценивает перенос embedding на новые классы.'),
       ('Почему нужно исключать self-match даже у frozen DINOv2?', 'Идентичная картинка с собой даст тривиально высокий score; это утечка протокола поиска, не вопрос обучения backbone.')],path=HOMEWORK)

    T('metric-pipeline','CNN с нуля против frozen DINOv2: полный retrieval pipeline',[15,16,19,20,21,22,23,24,27,29,30,31],
      'Сравнение retrieval-моделей фиксирует одинаковые данные и метрики, но различает supervised обучение и frozen extraction.',
      'CNN encoder обучается с triplet loss на train identities и выбирается по validation retrieval. Frozen DINOv2 выдаёт raw CLS без обучения новой головы. Обе модели затем нормируют embeddings и индексируют одну и ту же gallery. Exact Flat оценивает encoder; HNSW отдельно оценивает цену approximation. Checkpoint retrieval сравнивают на одинаковых query IDs. В коде training loop есть дополнительные приёмы SWA/scheduler; они не меняют определение semantic metrics и требуют отдельного контроля состояния.',
      r'x\xrightarrow{f_\theta}z\xrightarrow{L2}\widehat z\xrightarrow{\rm index}\mathrm{ranked\ IDs}\xrightarrow{\rm labels}\mathrm{metrics}',
      'Train CNN и frozen extraction — разные стадии. В retrieval loss не участвуют test labels; labels нужны только итоговой оценке ранжирования.',
      'Две модели получают те же144 gallery и96 query. Flat mAP сравнивает semantic features; HNSW ANN recall сравнивает индексы при фиксированных features.',
      ['Фиксируем split, preprocessing каждой модели и веса выбранного checkpoint.',
       'Извлекаем gallery/query embeddings, нормируем и получаем exact ranking.',
       'Отдельно строим HNSW на тех же vectors: изменение efSearch не требует дообучать encoder.'],
      'Нельзя объяснять плохой semantic mAP только HNSW, не проверив exact baseline.',[
       ('Что обучается в raw frozen DINOv2 baseline ДЗ7?', 'Ничего в backbone и никакая новая supervised-голова: CLS-features используют напрямую.'),
       ('Как отделить слабый encoder от ошибки ANN?', 'Сначала измерить semantic metrics exact Flat. Затем ANN recall относительно exact на тех же vectors.'),
       ('Требует ли смена efSearch повторного обучения CNN?', 'Нет. Это query-time параметр индекса при фиксированных embeddings.',*S(['Нет','Всегда да','Только если labels отрицательные','Всегда нужно менять patch size'])),
       ('Почему нельзя выбирать CNN checkpoint по test mAP и затем называть test независимым?', 'Test уже повлиял на выбор модели. Для выбора нужен validation, test используют после фиксации выбора.')],path=HOMEWORK)

    T('metric-cache','Кеш frozen features: IDs, labels и preprocessing',[23,24,25,27],
      'Кеширование сохраняет frozen embeddings, чтобы повторять оценку и поиск без нового forward backbone.',
      'Вместе с features сохраняют порядок image IDs и labels, checkpoint/revision и preprocessing. Матрица без этого контекста может оказаться несовместимой с текущей gallery. Frozen encoder извлекает признаки в eval и без autograd; для одинакового кеша нужны одинаковые входные views. Смена checkpoint или crop требует пересчёта кеша. Порядок строк features и labels должен совпадать даже при shuffle DataLoader.',
      r'F\in\mathbb R^{N\times D},\quad |\mathrm{IDs}|=|y|=N,\quad V_{float32}=4ND',
      'Считаем только feature payload, без labels, IDs и metadata. Для 240 изображений с D=384 это368640 байт.',
      'Validation gallery144 и query96 вместе дают240 строк. При D=384 float32 features занимают368640 байт до служебных данных.',
      ['Фиксируем checkpoint и transforms, включаем eval и отключаем граф.',
       'В одном и том же порядке собираем features, labels и image IDs.',
       'Сохраняем кеш и metadata; при смене preprocessing помечаем его устаревшим.'],
      'Совпадение формы старого кеша не доказывает совместимость весов, порядка объектов и preprocessing.',[
       ('240 vectors, D=384, float32. Сколько байт feature payload?', '368640.',*N(368640,r'240\cdot384\cdot4=368640',0)),
       ('Почему features.npy недостаточно без IDs и preprocessing?', 'Нельзя гарантировать соответствие строк объектам и воспроизвести смысл представлений.'),
       ('Можно ли сохранить labels отдельно в исходном порядке, если features извлекались shuffled loader?', 'Только при восстановленном совпадающем порядке. Иначе строки features и labels описывают разные объекты.'),
       ('Нужно ли пересчитывать кеш при смене checkpoint backbone?', 'Да, если нет доказанного эквивалентного представления; прежние features относятся к другим весам.')],path=HOMEWORK)

    # API cells use tiny synthetic fixtures; none completes a homework submission pipeline.
    def C(topic,key,lib,focus,prompt,setup,code,result,why,trap,test,doc=''):
        t=bank[topic]
        c=dict(id='l7-'+key,library=lib,topics=[topic],lesson=7,collection='code',focus=focus,prompt=prompt,
               setup=setup,code=code,result=result,explanation=why,trap=trap,test=test,runtime='cpu',
               source=deepcopy(t['source']),basis='Самостоятельный учебный пример на синтетических данных по уроку7; не решение ДЗ.',
               doc=doc or ('https://numpy.org/doc/stable/reference/index.html' if lib=='NumPy' else 'https://docs.pytorch.org/docs/2.14/index.html'))
        cards.append(c);t['codeCards'].append(c['id'])
        imports='import numpy as np\n' if lib in ('NumPy','FAISS') else 'import torch\nfrom torch import nn\nimport torch.nn.functional as F\n'
        if 'implementation' not in t:t['implementation']=dict(language='python',code=imports+setup+'\n\n'+code,note=why+' '+result,source=deepcopy(t['source']))
        questions.append(dict(id='api-'+c['id'],lesson=7,topic=topic,collection='code',type='open',level='apply',
            prompt=focus+': '+prompt+'\n\nДано:\n```python\n'+setup+'\n```',
            answer='```python\n'+code+'\n```\n\n'+why+'\n\nОжидаемый результат: '+result,example=trap,
            rubric=['Правильно выбрана операция и аргументы','Объяснена форма/значение результата','Учтено ограничение примера'],source=deepcopy(t['source']),codeCard=c['id']))

    C('ssl-simclr','positives','PyTorch','torch.arange: positive IDs','Для concatenation [view1 batch, view2 batch] постройте positive IDs.',
      'n = 3','positive = (torch.arange(2*n) + n) % (2*n)',
      '[3,4,5,0,1,2].','Вторая view исходника смещена на N, переход выполняется по модулю2N.',
      'Для interleaved layout [a1,a2,b1,b2] формула другая; нельзя переносить индексы между layout.',
      'assert positive.tolist() == [3,4,5,0,1,2]', 'https://docs.pytorch.org/docs/2.14/generated/torch.arange.html')
    C('ssl-simclr','self-mask','PyTorch','Tensor.fill_diagonal_','Исключите self-logits из softmax.',
      'logits = torch.eye(4)','logits.fill_diagonal_(float("-inf"))',
      'На диагонали−∞, вне её0.','Self больше не получает probability mass.',
      'Не закрывайте весь row; softmax строки из одних−∞ даёт NaN.',
      'assert torch.isneginf(logits.diag()).all()\nassert torch.allclose(logits.softmax(1).sum(1), torch.ones(4))')
    C('ssl-byol','ema','PyTorch','Tensor.lerp_: EMA','Обновите target=.9·target+.1·online без autograd.',
      'target = torch.tensor([2.])\nonline = torch.tensor([4.], requires_grad=True)',
      'with torch.no_grad():\n    target.lerp_(online, 0.1)',
      'Target=2.2, requires_grad=False.','lerp_(end,w) вычисляет (1−w)·self+w·end.',
      'Вес lerp равен1−momentum; здесь не.9.',
      'torch.testing.assert_close(target, torch.tensor([2.2]))\nassert not target.requires_grad')
    C('ssl-beit','masked-ce','PyTorch','F.cross_entropy: selected patches','Посчитайте CE только по двум masked-позициям.',
      'logits = torch.zeros(4, 8, requires_grad=True)\ntargets = torch.tensor([2, 5, 1, 7])\nmasked = torch.tensor([False, True, False, True])',
      'loss = F.cross_entropy(logits[masked], targets[masked])',
      'Loss=ln8, градиент только в выбранных строках.','Индексирование задаёт позиции, на которых определена цель.',
      'Mask позиций не является target vocabulary mask.',
      'loss.backward()\nassert torch.all(logits.grad[~masked] == 0)\nassert abs(loss.item()-float(np.log(8))) < 1e-6')
    C('ssl-ibot','soft-ce','PyTorch','F.log_softmax: soft target CE','Сравните student logits с detached teacher distribution.',
      'teacher = torch.tensor([[0.25, 0.75]], requires_grad=True)\nstudent = torch.zeros(1, 2, requires_grad=True)',
      'loss = -(teacher.detach() * F.log_softmax(student, dim=-1)).sum(-1).mean()',
      'Loss≈.693147, teacher.grad отсутствует.','Soft target задаёт вес каждого class log-probability.',
      'Передавать detached teacher logits как вероятности без softmax неверно.',
      'loss.backward()\nassert teacher.grad is None and student.grad is not None\nassert abs(loss.item()-float(np.log(2))) < 1e-6')
    C('ssl-dino','center','PyTorch','Tensor.mean: center statistics','Вычислите центр teacher logits по batch-оси.',
      'teacher_logits = torch.tensor([[1., 3.], [3., 5.]], requires_grad=True)',
      'batch_center = teacher_logits.detach().mean(dim=0, keepdim=True)',
      'Center shape[1,2], значения[2,4].','Усредняется batch, размерность выходной головы сохраняется.',
      'Mean по feature-оси дал бы другое состояние, непригодное для покомпонентного centering.',
      'torch.testing.assert_close(batch_center, torch.tensor([[2., 4.]]))\nassert not batch_center.requires_grad')
    C('ssl-dinov2','cls','PyTorch','Tensor indexing: CLS','Извлеките первый token каждого изображения как CLS-feature.',
      'hidden = torch.arange(2*5*3).reshape(2, 5, 3)',
      'cls = hidden[:, 0, :]',
      'Shape[2,3], строки[0,1,2] и[15,16,17].','Batch-ось сохраняется, token-ось выбирается индексом0.',
      'Этот пример предполагает CLS на позиции0 по конфигурации checkpoint.',
      'assert cls.tolist() == [[0,1,2],[15,16,17]]')
    C('ssl-dinov3','gram','PyTorch','torch.matmul: patch Gram','Вычислите попарные cosine similarities нормированных patches.',
      'features = F.normalize(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]), dim=1)',
      'gram = features @ features.T',
      'Матрица3×3 с единичной диагональю.','Строка — patch, поэтому F·Fᵀ даёт patch-to-patch отношения.',
      'Fᵀ·F имело бы форму2×2 по каналам.',
      'assert gram.shape == (3,3)\ntorch.testing.assert_close(gram.diag(), torch.ones(3))\nassert gram[0,1] == 0')
    C('ssl-dense','grid','NumPy','ndarray.reshape: spatial grid','Верните12 spatial tokens в сетку3×4.',
      'patches = np.arange(12*2).reshape(12, 2)',
      'grid = patches.reshape(3, 4, 2)',
      'Shape[3,4,2], grid[1,2] равен patches[6].','При row-major порядке плоский индекс row·width+col.',
      'Перед reshape удаляют только те special tokens, которые действительно есть в модели.',
      'assert grid.shape == (3,4,2) and np.array_equal(grid[1,2], patches[6])')
    C('ssl-probe','freeze','PyTorch','Module.requires_grad_','Заморозьте все параметры backbone, сохранив обучаемый head.',
      'backbone = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5))\nhead = nn.Linear(8, 3)\nbackbone.eval()',
      'backbone.requires_grad_(False)',
      'Параметры backbone не требуют grad, параметры head требуют.','requires_grad_ меняет trainability рекурсивно.',
      'eval установлен отдельно: requires_grad_ не управляет Dropout.',
      'assert all(not p.requires_grad for p in backbone.parameters())\nassert all(p.requires_grad for p in head.parameters())')
    C('metric-calibration','reject','NumPy','numpy.where: rejection','Примените фиксированный threshold=.5 к maximum score.',
      'scores = np.array([[0.4, 0.35], [0.2, 0.8]])\nthreshold = 0.5',
      'predictions = np.where(scores.max(axis=1) >= threshold, scores.argmax(axis=1), -1)',
      '[-1,1].','Maximum определяет принятие, argmax — ответ среди известных классов.',
      'Threshold уже выбран на validation; этот вызов его не обучает.',
      'assert predictions.tolist() == [-1,1]')
    C('metric-tasks','cosine','PyTorch','F.cosine_similarity','Сравните пару feature-векторов по cosine.',
      'a = torch.tensor([[3., 4.]])\nb = torch.tensor([[1., 0.]])',
      'score = F.cosine_similarity(a, b, dim=1)',
      'Score=.6.','Скалярное произведение3 делится на нормы5 и1.',
      'Больший cosine означает более близкую пару, в отличие от distance.',
      'torch.testing.assert_close(score, torch.tensor([0.6]))')
    C('retrieval-recall','recall','NumPy','numpy.count_nonzero: semantic recall','Для одного query вычислите долю positives, найденных в top2.',
      'gallery_labels = np.array([4, 2, 4, 4, 4])\nranking = np.array([1, 0, 2, 3, 4])\nlabel = 4\nk = 2',
      'recall = np.count_nonzero(gallery_labels[ranking[:k]] == label) / np.count_nonzero(gallery_labels == label)',
      'Recall=.25.','Один найденный positive делится на4 positives всей gallery.',
      'Знаменатель k дал бы precision=.5.',
      'assert recall == 0.25')
    C('retrieval-map','ap','NumPy','numpy.cumsum: AP','Рассчитайте полный AP по заданной binary relevance.',
      'relevant = np.array([True, False, True])\nranks = np.arange(1, len(relevant)+1)',
      'precision = np.cumsum(relevant) / ranks\nap = precision[relevant].sum() / relevant.sum()',
      'AP=5/6.','В сумму входят precision только на relevant-позициях.',
      'Полное ранжирование и наличие хотя бы одного relevant здесь заданы fixture.',
      'assert abs(ap-5/6) < 1e-12')
    C('retrieval-ndcg','dcg','NumPy','numpy.log2: DCG','Посчитайте DCG@2 для graded relevance[0,1].',
      'relevance = np.array([0., 1.])\nranks = np.arange(1, 3)',
      'dcg = ((2**relevance - 1) / np.log2(ranks + 1)).sum()',
      'DCG≈.63093.','Discount использует rank+1; для первого rank1 log2(2)=1.',
      'Это ещё не NDCG: для него нужно IDCG.',
      'assert abs(dcg-1/np.log2(3)) < 1e-12')
    C('metric-verification','far','NumPy','numpy.mean: impostor acceptance','Вычислите FAR по score только impostor-пар.',
      'impostor_scores = np.array([0.1, 0.8, 0.2, 0.3])\nthreshold = 0.7',
      'far = (impostor_scores >= threshold).mean()',
      'FAR=.25.','Одна из четырёх отрицательных пар ошибочно принята.',
      'Если смешать genuine в тот же знаменатель, это перестанет быть FAR.',
      'assert far == 0.25')
    C('metric-triplet','hinge','PyTorch','torch.relu: triplet hinge','Примените margin hinge к готовым distances.',
      'd_positive = torch.tensor([0.4, 0.2])\nd_negative = torch.tensor([0.5, 0.8])\nmargin = 0.2',
      'losses = torch.relu(d_positive - d_negative + margin)',
      '[.1,0].','Hinge не штрафует тройку, уже удовлетворяющую margin.',
      'Здесь d — обычное расстояние, не squared distance.',
      'torch.testing.assert_close(losses, torch.tensor([0.1, 0.]))')
    C('metric-mining','positive-mask','PyTorch','torch.eye: exclude self','Постройте same-class positive mask без диагонали.',
      'labels = torch.tensor([0, 0, 1, 1])\nsame = labels[:,None] == labels[None,:]',
      'positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)',
      'У каждого anchor один positive, диагональ False.','Диагональная маска отдельно исключает self.',
      'Не используйте positive=~negative без удаления диагонали.',
      'assert not positive.diag().any() and positive.sum().item() == 4')
    C('metric-mining','nearest-negative','PyTorch','Tensor.masked_fill: nearest negative','Выберите ближайший negative по готовой строке distance.',
      'distances = torch.tensor([0., 0.7, 0.6, 1.1])\nnegative = torch.tensor([False, False, True, True])',
      'nearest = distances.masked_fill(~negative, float("inf")).min()',
      'Nearest negative=.6.','Запрещённые значения не могут выиграть min.',
      'Для hardest positive нужен max и противоположный знак infinity.',
      'torch.testing.assert_close(nearest, torch.tensor(0.6))')
    C('metric-pk','class-counts','NumPy','numpy.unique: P×K check','Проверьте количество представленных классов и объектов каждого.',
      'labels = np.array([4,4,4,9,9,9])',
      'classes, counts = np.unique(labels, return_counts=True)',
      'Два класса, counts[3,3]: P=2,K=3.', 'Counts проверяет фактический состав готового batch.',
      'Проверка не создаёт sampler и не гарантирует разные image IDs внутри класса.',
      'assert len(classes) == 2 and counts.tolist() == [3,3]')
    C('face-normalized','cosine-logits','PyTorch','F.linear: normalized classifier','Вычислите scaled cosine logits для нормированных feature и weights.',
      'features = F.normalize(torch.tensor([[3., 4.]]), dim=1)\nweight = F.normalize(torch.eye(2), dim=1)\nscale = 10.',
      'logits = scale * F.linear(features, weight)',
      'Logits[[6,8]].','linear без bias вычисляет feature·weightᵀ.',
      'В произвольном classifier без нормировки это не cosine.',
      'torch.testing.assert_close(logits, torch.tensor([[6.,8.]]))')
    C('face-margins','arcface-angle','PyTorch','torch.cos: ArcFace target','Вычислите иллюстративный ArcFace target cosine при θ=π/3,m=π/6.',
      'import math\ntheta = torch.tensor([math.pi/3])\nmargin = math.pi/6',
      'target_cosine = torch.cos(theta + margin)',
      'Приблизительно0.','Margin прибавляется к углу, не вычитается из cosine.',
      'Это одна геометрическая операция, не полный production ArcFace head с численными guards.',
      'assert abs(target_cosine.item()) < 1e-6')
    C('face-adaface','quality-proxy','PyTorch','Tensor.detach: quality norm','Стандартизуйте detached pre-normalization norms и ограничьте их.',
      'norms = torch.tensor([5., 10., 20.], requires_grad=True)\nmean, std, h = 10., 5., 0.5',
      'quality = ((norms.detach() - mean) / std * h).clamp(-1, 1)',
      '[-.5,0,1], без grad.','Стандартизация использует исходные norms и заданные running statistics.',
      'В реальном обучении mean/std обновляются по согласованному правилу и std защищают от0.',
      'torch.testing.assert_close(quality, torch.tensor([-0.5,0.,1.]))\nassert not quality.requires_grad')
    C('search-flat','normalize','NumPy','numpy.linalg.norm: row normalize','Нормируйте features по строкам без NaN для нулевой строки.',
      'vectors = np.array([[3.,4.],[0.,0.]], dtype=np.float32)',
      'normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)',
      '[[.6,.8],[0,0]], float32.','keepdims сохраняет [N,1] для корректного broadcasting.',
      'Нулевая строка остаётся нулём и всё ещё не имеет математически определённого cosine.',
      'assert normalized.dtype == np.float32 and np.isfinite(normalized).all()\nnp.testing.assert_allclose(normalized, [[.6,.8],[0,0]], atol=1e-7)')
    C('search-flat','flat-ip','FAISS','faiss.IndexFlatIP','Создайте exact inner-product index для двух нормированных vectors.',
      'import faiss\ngallery = np.array([[1.,0.],[0.,1.]], dtype=np.float32)',
      'index = faiss.IndexFlatIP(2)\nindex.add(gallery)',
      'Index содержит2 vectors, train не нужен.', 'Flat хранит исходные vectors и сравнивает со всей gallery.',
      'IndexFlatIP не нормирует входы автоматически.',
      'scores, ids = index.search(np.array([[1.,0.]], dtype=np.float32), 2)\nassert index.ntotal == 2 and ids.tolist() == [[0,1]]\nnp.testing.assert_allclose(scores, [[1.,0.]])',FAISS+'Faiss-indexes')
    C('search-hnsw','hnsw','FAISS','faiss.IndexHNSWFlat','Создайте маленький HNSW с inner-product metric и задайте efSearch.',
      'import faiss\ngallery = np.array([[1.,0.],[0.,1.],[-1.,0.]], dtype=np.float32)',
      'index = faiss.IndexHNSWFlat(2, 8, faiss.METRIC_INNER_PRODUCT)\nindex.hnsw.efSearch = 32\nindex.add(gallery)',
      'Граф содержит3 vectors; query[1,0] находит ID0.', 'Metric совпадает с exact IP baseline, efSearch управляет query effort.',
      'Точность на трёх vectors не доказывает высокий recall на большой коллекции.',
      'scores, ids = index.search(np.array([[1.,0.]], dtype=np.float32), 1)\nassert ids[0,0] == 0 and index.ntotal == 3',FAISS+'Faiss-indexes')
    C('search-ivfpq','pq-train','FAISS','IndexIVFPQ.train','Обучите маленький учебный IVF-PQ codebook до add.',
      'import faiss\nfaiss.omp_set_num_threads(1)\ntrain_vectors = np.random.default_rng(7).normal(size=(512,8)).astype(np.float32)\nindex = faiss.IndexIVFPQ(faiss.IndexFlatL2(8), 8, 4, 2, 2)',
      'index.train(train_vectors)',
      'is_trained=True, ntotal=0.', 'Обучаются coarse/PQ codebooks, сами vectors ещё не добавлены.',
      'В большом проекте codebook train data должны представлять distribution коллекции.',
      'assert index.is_trained and index.ntotal == 0\nindex.add(train_vectors[:10])\nassert index.ntotal == 10',FAISS+'Faiss-indexes')
    C('search-lsh','sign-hash','NumPy','numpy.matmul: hyperplane bits','Получите сигнатуры знаков относительно двух заданных гиперплоскостей.',
      'vectors = np.array([[1.,2.],[-1.,2.]])\nplanes = np.eye(2)',
      'bits = vectors @ planes.T >= 0',
      '[[True,True],[False,True]].', 'Каждый бит — сторона соответствующей гиперплоскости.',
      'В примере planes заданы вручную; для random-hyperplane LSH их выбирают случайно по нужному распределению.',
      'assert bits.tolist() == [[True,True],[False,True]]')
    C('metric-protocol','split-overlap','NumPy','numpy.intersect1d: split audit','Проверьте пересечение class IDs train и validation.',
      'train_classes = np.array([0,2,4])\nvalidation_classes = np.array([1,3])',
      'overlap = np.intersect1d(train_classes, validation_classes)',
      'Пустой массив.', 'Проверяются class IDs, а не только image IDs.',
      'Для полного протокола также проверяют test classes и отсутствие повторов gallery/query images.',
      'assert overlap.size == 0')
    C('metric-pipeline','rank','NumPy','numpy.argsort: exact cosine ranking','Ранжируйте готовую строку similarity по убыванию.',
      'scores = np.array([0.2, 0.9, -0.1, 0.7])',
      'ids = np.argsort(-scores, kind="stable")',
      '[1,3,0,2].', 'Больший similarity должен идти первым.',
      'Для distances знак другой; ties требуют согласованного правила.',
      'assert ids.tolist() == [1,3,0,2]')
    C('search-hnsw','ann-overlap','NumPy','numpy.intersect1d: ANN recall','Сравните exact и approximate top3 одного query.',
      'exact = np.array([1,2,3])\napproximate = np.array([1,3,8])',
      'ann_recall = np.intersect1d(exact, approximate).size / len(exact)',
      '2/3.', 'Совпадение относится к IDs, не class labels.',
      'Top-K должны иметь одинаковый K и корректные уникальные IDs.',
      'assert abs(ann_recall-2/3) < 1e-12')
    C('search-flat','flat-l2','FAISS','faiss.IndexFlatL2: squared distance','Найдите squared L2 от query0 до gallery[2,0].',
      'import faiss\nindex = faiss.IndexFlatL2(2)\nindex.add(np.array([[2.,0.]], dtype=np.float32))',
      'distances, ids = index.search(np.array([[0.,0.]], dtype=np.float32), 1)',
      'Distance4, ID0.', 'FAISS L2 возвращает квадрат евклидовой нормы.',
      'Обычное Euclidean distance равно2, не4.',
      'assert distances[0,0] == 4 and ids[0,0] == 0',FAISS+'MetricType-and-distances')

    C('metric-cache','cache-copy','NumPy','ndarray.copy: frozen snapshot','Сделайте независимую копию готового массива features.',
      'features = np.array([[1.,2.],[3.,4.]], dtype=np.float32)',
      'snapshot = features.copy()',
      'Snapshot не меняется при перезаписи исходного массива.', 'Копия отделяет сохранённый snapshot от изменяемого рабочего буфера.',
      'Copy не сохраняет файл на диск и не добавляет metadata о checkpoint.',
      'features[0,0] = 99\nassert snapshot[0,0] == 1 and not np.shares_memory(snapshot, features)')

    def D(key,topic,title,summary,steps):
        t=bank[topic]
        nodes=[dict(id=str(i),col=1,row=i,title=title_,hint=hint,detail=detail,
                    workedExample=worked,kind='input' if i==0 else 'output' if i==len(steps)-1 else 'block')
               for i,(title_,hint,detail,worked) in enumerate(steps)]
        d=dict(id=key,topic=topic,lesson=7,title=title,summary=summary,scenario=t['example'],source=deepcopy(t['source']),
               nodes=nodes,edges=[dict(**{'from':str(i),'to':str(i+1)},label='',kind='flow') for i in range(len(nodes)-1)],groups=[])
        diagrams.append(d);t['diagrams'].append(key)

    D('l7-simclr','ssl-simclr','SimCLR: от views к NT-Xent',
      'Показан учебный pipeline двух views; encoder и projector общие между ветвями. Матрица loss объединяет все views батча.',[
       ('Изображения → 2 views','N → 2N картинок','Две случайные аугментации исходника задают positive pair.','N=3 даёт6 views. Важно сохранить ID исходника, чтобы target positive не перепутался с другой картинкой.'),
       ('Общий encoder f','2N×D признаков h','Одинаковые веса encoder применяются ко всем views; это две обработки, а не две независимые модели.','Шесть картинок превращаются в h формы[6,8] в учебном примере. Веса f общие, градиенты приходят от всех views.'),
       ('Projection head g','h → z → L2','MLP переносит h в contrastive space; z нормируют для cosine.','Head8→4 создаёт z[6,4]. Нормируем каждую строку, не всю матрицу одной общей нормой.'),
       ('Similarity / temperature','6×6 logits','Скалярные произведения views делятся на τ; self-logits закрыты.','Из36 scores шесть диагональных закрыты. В каждой строке5 разрешённых кандидатов, среди которых один positive.'),
       ('Positive targets → CE','NT-Xent по всем anchors','Cross-entropy повышает вероятность правильной второй view среди кандидатов.','Для layout[a1,b1,c1,a2,b2,c2] targets=[3,4,5,0,1,2]. Усредняем шесть направленных losses.'),
       ('Downstream features h','обычно до head','Для новой задачи берут encoder и выбирают протокол frozen probe или fine-tuning.','После предобучения g можно убрать и использовать h[1,8] для новой картинки. Это ещё не готовый classifier labels.')])
    D('l7-byol','ssl-byol','BYOL: две ветви и разные обновления',
      'Схема развёрнута в последовательность логических стадий; online и target обрабатывают разные views параллельно. EMA относится к следующему состоянию teacher.',[
       ('Две views','один исходник','Одна view идёт в online, другая в target; loss также симметризуют сменой ролей.','Из картинки получаем cropA и cropB. Они должны сохранять полезную семантику, хотя пиксели и размеры crop отличаются.'),
       ('Online f + g + q','encoder/projector/predictor','Predictor есть только в online-пути.','Online выдаёт q[1,4]; три компонента f,g,q получают градиенты от сравнения с target другой view.'),
       ('Target f + g','отдельные EMA параметры','Target не имеет online predictor и не обновляется обычным optimizer.','Target выдаёт z_t[1,4] в отдельной ветви; это не ссылка на тот же online Parameter, а собственная копия весов.'),
       ('L2 + stop-gradient','target без autograd','Сравнивают нормированные векторы, teacher detached.','При loss=(q_norm−z_target_norm)² student получает производную, а teacher.grad остаётся None.'),
       ('Optimizer → online','градиентный шаг','Обновляет только online-параметры.','Если после optimizer некоторый online-параметр стал4, это новое значение пойдёт в EMA. Target ещё может оставаться2.'),
       ('EMA → next target','τθt+(1−τ)θo','Смешивание выполняется без градиента и изменяет состояние target для следующего шага.','Для τ=.9 target2 и online4 получаем2.2. Мгновенная замена на4 была бы другой динамикой.')])
    D('l7-masked','ssl-ibot','Masked prediction: BEiT и iBOT',
      'Общая учебная схема; источник target принципиально различается: внешний dVAE у BEiT и EMA teacher у iBOT.',[
       ('Изображение → patches','сохраняем геометрию','Нумерация патчей должна соответствовать входу и targets.','Четыре spatial patches получают индексы0,1,2,3. При смене crop одинаковый индекс может указывать уже на другую область.'),
       ('Mask выбранных позиций','M={1,3}','Student получает повреждённые позиции с mask embedding.','Скрываем входы1 и3, но targets этих позиций оставляем в отдельном массиве для loss; не раскрываем правильные коды во входе.'),
       ('Источник target','BEiT dVAE / iBOT teacher','BEiT использует discrete codes, iBOT — online teacher distributions.','В BEiT целями могут быть IDs5 и7 из vocabulary8. В iBOT это, например, две soft distributions; цели не взаимозаменяемы.'),
       ('Student ViT + head','logits по позициям','Student предсказывает target выбранных пространственных позиций.','При двух masked positions и vocabulary8 head выдаёт[2,8] logits. Для iBOT нужно согласовать также области crops.'),
       ('Masked objective','только выбранные позиции','Loss берёт CE дискретного или soft target согласно методу.','Открытые позиции не получают именно masked-loss этой задачи; дополнительные image-level losses считаются отдельно по своему правилу.'),
       ('Обновление / features','teacher EMA у iBOT','Student учится градиентом; внешний tokenizer BEiT не превращается из-за этого в EMA teacher.','В iBOT teacher обновляется EMA student. В BEiT pretrained tokenizer задаёт фиксированный источник codes в этом изложении.')])
    D('l7-dino','ssl-dino','DINO: global/local, center и EMA',
      'Student и teacher имеют разные наборы views. Center logits и EMA weights показаны отдельными состояниями; обе операции выполняются без train-градиента teacher.',[
       ('Global и local crops','2 global + 4 local','Teacher получает global, student — все views.','В учебном batch одной картинки teacher выполняет2 forward, student6 forward. Нельзя просто объявить все8 outputs независимыми negatives.'),
       ('Student / teacher heads','распределения по K','Выходная голова задаёт пространство self-distillation, не обязательные ручные semantic labels.','При K=8 каждая view имеет8 logits. Student и teacher могут выдавать разные числа до преобразования в распределения.'),
       ('Teacher center + softmax','stop-grad / temperature','Teacher logits центрируются, target detached; student использует log-softmax.','Из teacher logits вычитаем center[1,8] и делим на teacher temperature. Center не является размером learning rate или class ID.'),
       ('Cross-view CE','исключаем same view','Сравниваются teacher global и несовпадающие student views.','Из2×6=12 пар убираем2 self-view сравнения. Осталось10 CE, в том числе global-to-local.'),
       ('Student optimizer','градиент только student','Teacher не получает шаг этого optimizer.','Backward по среднему10 losses обновляет student. Target распределения teacher остаются константами внутри данного шага.'),
       ('Teacher EMA / center EMA','разные состояния','Teacher weights смешивают со student, center — со статистиками teacher logits.','Teacher-параметр может стать2.2 из2 и4 приτ=.9. Center при этом имеет форму[1,K] и обновляется по совершенно другим данным.')])
    D('l7-arcface','face-margins','ArcFace: обучающая голова и inference',
      'Показан target-margin во время обучения. На inference labels и classifier centers не обязательны: используется embedding и метрика сравнения.',[
       ('Image → encoder','feature z','Encoder преобразует картинку в feature; его норма ещё может не равняться1.','Учебный выход z=[3,4] имеет норму5. Это представление одного изображения, а не probability distribution классов.'),
       ('Нормировка z и W','unit hypersphere','Нормируем feature и class centers по feature dimension.','Из[3,4] получаем[.6,.8]. Строки W также нормируем: иначе logits зависели бы ещё от нормы центра.'),
       ('Cosine logits','cosθj','Скалярные произведения с центрами задают угловую близость.','Для центров осей cosines=[.6,.8]. Число logits равно числу training identities, а не размерности самого embedding.'),
       ('Target angular margin','cos(θy+m)','Меняем только logit известного target-класса.','При θy=60° и m=30° target cosine0 вместо.5. Non-target cosine.2 остаётся.2 до scale.'),
       ('Scale s → CE','только обучение','Масштабируем logits и применяем class-label CE.','Scale10 превращает target0 и non-target.2 в logits0 и2. CE заставляет encoder сделать target менее неоднозначным.'),
       ('Inference embedding','cosine / threshold','Margin head больше не требует label новой картинки.','Для новой identity берём нормированный z и сравниваем с gallery. Центры training classes не обязаны присутствовать в inference pipeline.')])
    D('l7-retrieval','metric-pipeline','Retrieval: encoder, gallery, query и оценка',
      'Semantic evaluation и ANN approximation разделены: точный поиск оценивает features, приближённый — качество ускорения относительно того же baseline.',[
       ('Train / val / test IDs','class-disjoint split','Train identities не пересекаются с validation/test identities.','В ДЗ7 это12/6/6 классов. Validation gallery144 и query96 не содержат одинаковых image IDs между собой.'),
       ('Encoder → features','CNN или frozen CLS','CNN обучается на train, frozen DINOv2 использует pretrained CLS без новой головы.','Обе модели извлекают свои vectors для одних и тех же144 gallery images и96 queries. Их feature dimensions могут различаться.'),
       ('L2-normalize → Flat','exact baseline','Нормируем обе стороны и получаем ranking cosine similarity.','Матрица scores имеет96×144 элементов. Для полного AP нужны все144 позиции ранжирования каждого query.'),
       ('Labels → semantic metrics','Recall / Hit / mAP','Semantic relevance определяется class labels.','У query24 positives. Даже правильный top1 даёт Recall@1=1/24, Hit@1=1; смешивать эти числа нельзя.'),
       ('HNSW → ANN recall','те же vectors и metric','Сравниваем ID approximate neighbors с exact top-K.','Если top10 HNSW содержит8 exact neighbors, ANN recall=.8 независимо от semantic labels этих восьми изображений.'),
       ('Сравнение / latency','одинаковый протокол','Выбор checkpoint по validation; финальный test после фиксации настроек.','Можно увеличить efSearch и увидеть рост времени и recall. Если Flat mAP низок, настройка HNSW не исправляет сам encoder.')])
    D('l7-ivfpq','search-ivfpq','IVF-PQ: обучение индекса и поиск',
      'Это архитектура индекса, а не нейросети. Две причины ошибки — coarse candidate selection и quantization distances — требуют разных проверок.',[
       ('Train vectors','репрезентативный пул','Обучающая коллекция для coarse/PQ codebooks должна отражать distribution.','Вектор D=128 делим на16 частей по8 координат. Train нужен индексу даже при полностью frozen image encoder.'),
       ('Coarse centroids / lists','nlist','Каждый gallery vector назначается в coarse list.','При nlist=256 запрос может просмотреть только nprobe=8 списков. Не выбранный список может содержать истинного ближайшего соседа.'),
       ('PQ subcodes','M частей × b bits','Подпространства кодируются обученными центрами.','M16,b8 даёт16 байт vector code без ID. Исходный float32 vector D128 занимал512 байт до сжатия.'),
       ('Query → nprobe lists','coarse candidate pool','Search выбирает ограниченный пул кандидатов.','Увеличение nprobe с8 до32 расширяет пул и обычно увеличивает время. Это не обучает заново encoder или PQ codebooks.'),
       ('Approximate distances','ranking по кодам','Расстояния оцениваются по quantized representation.','Даже просмотр всех256 списков сохраняет погрешность PQ. Это объясняет, почему nprobe=nlist не всегда даёт exact order.'),
       ('Optional exact rerank','проверка кандидатов','Исходные vectors переупорядочивают выбранный shortlist.','Если нужный neighbor не попал в shortlist, rerank его не найдёт. Для rerank также нужна память/доступ к исходным vectors.')])

    corrections.extend([
        dict(lesson=7,topic='search-flat',title='Правильное имя FAISS-класса',text='В markdown ДЗ7 написано faiss.FlatIPIndex. В Python API используется faiss.IndexFlatIP; train для Flat не нужен.',url=FAISS+'Faiss-indexes'),
        dict(lesson=7,topic='metric-calibration',title='Known-only rejection и open-set',text='Демонстрация calibration/rejection делит только выбранные known classes. Для измерения принятия unknown нужны отдельные unknown-классы в протоколе; reject rate на known-only test этого не заменяет.',url='https://github.com/T-CV/cv-course-at-mipt-fall-2026/blob/5db8762/lesson_7/lecture.ipynb'),
    ])
    assert len(bank)==28
    assert all(t.get('implementation') and t['codeCards'] for t in bank.values())
    return dict(repository='https://github.com/T-CV/cv-course-at-mipt-fall-2026',revision='5db8762',date='2026-10-28',
                added=['lesson_7/lecture.ipynb','lesson_7/homework.ipynb'],topics=list(bank))
