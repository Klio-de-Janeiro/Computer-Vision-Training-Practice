"""Authored architecture graphs grounded in the existing lecture topics."""
from copy import deepcopy


def build(topics):
    lookup={t['id']:t for t in topics}
    diagrams=[]
    def V(key,col,row,title,hint,detail,kind='block'):
        return {'id':key,'col':col,'row':row,'title':title,'hint':hint,'detail':detail,'kind':kind}
    def D(key,topic,title,summary,nodes,edges,groups=()):
        diagrams.append({'id':key,'topic':topic,'lesson':lookup[topic]['lesson'],'title':title,
            'summary':summary,'nodes':nodes,
            'edges':[{'from':e[0],'to':e[1],'label':e[2] if len(e)>2 else '',
                      'kind':e[3] if len(e)>3 else 'flow'} for e in edges],
            'groups':[{'label':g[0],'col':g[1],'row':g[2],'cols':g[3],'rows':g[4]} for g in groups],
            'source':deepcopy(lookup[topic]['source'])})
    def C(key,topic,title,summary,steps):
        nodes=[V(str(i),1,i,*step) for i,step in enumerate(steps)]
        D(key,topic,title,summary,nodes,[(str(i),str(i+1)) for i in range(len(nodes)-1)])

    D('bert','bert','BERT: encoder и две задачи предобучения',
      'Исходный BERT: двунаправленный encoder, MLM на выбранных позициях и NSP по [CLS]. У BERT-Base L=12, у Large L=24. Показанный post-norm блок повторяется L раз.',[
        V('tokens',1,0,'Токены + спецтокены','[CLS], [SEP], [MASK]','WordPiece-последовательность включает специальные позиции. При MLM вход искажён, а targets хранят исходные токены.','input'),
        V('emb',1,1,'Сумма embeddings','token + position + segment','Обучаемые token, position и segment embeddings складываются. Далее dropout и нормировка согласно конфигурации BERT.'),
        V('msa',1,2,'Multi-head attention','двунаправленный self-attn','Каждая реальная позиция видит доступные токены с обеих сторон. Padding исключается маской.'),
        V('norm1',1,3,'Add + LayerNorm','residual после attention','Выход attention складывается с входом подслоя, затем нормируется: post-norm исходного BERT.'),
        V('ffn',1,4,'FFN','Linear → GELU → Linear','Один position-wise MLP применяется с общими весами ко всем позициям. В лекции промежуточная ширина равна 4d.'),
        V('norm2',1,5,'Add + LayerNorm','контекстные токены [B,T,d]','Второй residual и нормировка завершают encoder block. Следующий слой получает весь набор токенов.'),
        V('mlm',2,6,'MLM head','выбранные позиции → vocab','Предсказывает исходные токены выбранных MLM-позиций. Это token-level голова.','output'),
        V('nsp',0,6,'[CLS] → NSP head','IsNext / NotNext','В исходном предобучении pooled [CLS] используется для классификации пары предложений. Downstream head может быть другой.','output'),
        V('loss',1,7,'MLM + NSP loss','только обучение','Сумма двух обучающих целей исходного BERT. Для продолжения текста добавляют новую MASK-позицию; decoder-only генерация из этой схемы не следует.','loss')],
      [('tokens','emb'),('emb','msa'),('msa','norm1'),('emb','norm1','residual','skip'),('norm1','ffn'),('ffn','norm2'),('norm1','norm2','residual','skip'),('norm2','mlm'),('norm2','nsp'),('mlm','loss','train','training'),('nsp','loss','train','training')],
      [('ENCODER BLOCK × L',1,2,1,4)])

    D('attention','attention','Scaled dot-product attention',
      'Одна голова: scores определяют веса по ключам, затем этими весами агрегируются values. n — число queries, m — число keys/values.',[
        V('q',0,0,'Queries Q','[n, dk]','Один query задаёт одну выходную позицию. В cross-attention он может происходить из другой последовательности.','input'),
        V('k',1,0,'Keys K','[m, dk]','Keys сравниваются с каждым query. Число keys согласовано с числом values.','input'),
        V('v',2,0,'Values V','[m, dv]','Передаваемое содержимое. QK определяет выбор, а V — то, что будет смешано.','input'),
        V('scores',0,1,'QKᵀ / √dk','[n, m]','Масштабирование удерживает типичный разброс logits при стандартных предположениях о компонентах Q/K.'),
        V('mask',0,2,'Добавить маску','0 / −∞','Запрещённые пары получают −∞ до softmax. Полностью запрещённая строка требует отдельной обработки.'),
        V('softmax',0,3,'Softmax по keys','сумма строки = 1','Каждая строка становится распределением весов по допустимым ключам. Softmax не обязан сохранять ранг logits.'),
        V('weighted',1,4,'A · V','[n, dv]','Взвешенная сумма values даёт один выходной вектор на query.','output')],
      [('q','scores'),('k','scores'),('scores','mask'),('mask','softmax'),('softmax','weighted','веса'),('v','weighted','values','context')])

    D('encoder','heads','Transformer encoder: вариант pre-norm',
      'Базовый блок по компонентам лекции. Здесь показан pre-norm; исходный BERT на отдельной схеме использует post-norm.',[
        V('x',1,0,'Вход X','[B,N,d]','Позиционная информация задаётся выбранной моделью до/внутри attention.','input'),
        V('ln1',1,1,'LayerNorm','по признакам токена','Нормировка предшествует attention в показанном pre-norm варианте.'),
        V('attn',1,2,'Multi-head self-attn','Q,K,V из одной X','Головы смешивают информацию между позициями, их выходы конкатенируются и проецируются.'),
        V('add1',1,3,'Residual sum','X + Attention(LN(X))','Тождественный shortcut обходит подслой attention.'),
        V('ffn',1,4,'LayerNorm → FFN','d → dff → d','Position-wise MLP смешивает признаки каждой позиции; он сам не создаёт новых межпозиционных связей.'),
        V('add2',1,5,'Residual sum','выход блока','Второй shortcut обходит нормировку и FFN. Блок может повторяться L раз.','output')],
      [('x','ln1'),('ln1','attn'),('attn','add1'),('x','add1','residual','skip'),('add1','ffn'),('ffn','add2'),('add1','add2','residual','skip')])

    D('decoder','heads','Transformer decoder с cross-attention',
      'Структура encoder–decoder декодера: masked self-attention, чтение source memory, FFN. В decoder-only модели отдельного source cross-attention обычно нет.',[
        V('text',0,0,'Target tokens','сдвинутый префикс','На train входы известны через teacher forcing, но будущие target-позиции закрыты causal mask.','input'),
        V('memory',2,1,'Source memory','выход encoder','Полный известный source, например visual tokens изображения. Padding memory маскируется отдельно.','input'),
        V('self',0,1,'Causal self-attention','+ norm / residual','Target-позиция агрегирует свой доступный префикс.'),
        V('cross',0,2,'Cross-attention','Q: target; K,V: memory','Запросы относятся к target, ключи и значения — к source memory. Причинный порядок target не запрещает читать конец source.'),
        V('ffn',0,3,'FFN','+ norm / residual','Преобразует признаки после двух attention-подслоёв. Расположение нормировок зависит от варианта блока.'),
        V('head',0,4,'Vocabulary head','logits следующего токена','После стека decoder blocks линейная голова выдаёт scores по словарю. Выбор токена — отдельный шаг decoding.','output')],
      [('text','self'),('self','cross','Q'),('memory','cross','K,V','context'),('cross','ffn'),('ffn','head')])

    C('vit','vit','Vision Transformer (ViT)',
      'Классический ViT для классификации: один CLS, learned positions и стек encoder blocks. Число слоёв и ширина зависят от конфигурации.',[
        ('Изображение','[B,C,H,W]','Изображение разбивается на неперекрывающиеся патчи. В этой схеме H и W делятся на P.','input'),
        ('Patch projection','N = HW/P²; [B,N,d]','Flatten каждого патча и общая Linear-проекция эквивалентны Conv2d с kernel=stride=P при согласованном порядке координат.'),
        ('Добавить CLS и PE','[B,N+1,d]','CLS — обучаемый токен, не истинный класс. Позиционные embeddings задают положение патчей.'),
        ('Encoder blocks × L','LN → MSA; LN → MLP','В классическом ViT блоки pre-norm с residual-связями. Full attention связывает все токены, число пар квадратично по N.'),
        ('Final LayerNorm → CLS','[B,d]','Для классической классификации берётся итоговое представление CLS.'),
        ('Classification head','[B,число классов]','Линейная/MLP-голова выдаёт class logits. Другие задачи могут использовать patch tokens.','output')])

    C('swin','swin','Swin Transformer: окна и стадии',
      'Swin строит иерархию карт. Внутри стадии чередуются обычные и сдвинутые окна; patch merging меняет масштаб между стадиями.',[
        ('Изображение → patches','начальная сетка H₀×W₀','Patch partition и линейный embedding превращают пиксельные блоки в признаки.','input'),
        ('W-MSA block','локальные окна M×M','Оконное attention плюс norm, residual и MLP. В одном окне M² токенов текущей карты, а не обязательно M² исходных пикселей.'),
        ('SW-MSA block','сдвиг floor(M/2)','Сдвиг меняет оконные группы. При cyclic shift маска запрещает ложные связи краёв, затем выполняется обратный сдвиг. В блоке также есть MLP и residual.'),
        ('Повторить пару блоков','стадия: размер карты прежний','Чередование связывает прежние окна постепенно. Число блоков стадии задаёт конкретная конфигурация.'),
        ('Patch merging','H/2 × W/2 × 2C','Соседние 2×2 токена конкатенируются в 4C, нормируются и проецируются обычно в 2C.'),
        ('Следующие стадии','карты разных масштабов','Операции оконных блоков и merging повторяются. Локальные группы работают уже на более грубой spatial-сетке.'),
        ('Голова задачи','classification / detection','Классификация использует агрегированные признаки, dense-задачи могут получать несколько уровней backbone.','output')])

    for key,title,loss in [('clip','CLIP: два независимых encoder','Симметричная softmax CE'),('siglip','SigLIP: парный sigmoid objective','Pairwise sigmoid loss')]:
        D(key,key,title,
          'Image и text пути кодируются независимо. Совместное пространство позволяет заранее вычислять embeddings кандидатов. Различие этих схем — в обучающем objective, а не в добавлении caption decoder.',[
            V('image',0,0,'Изображение','resize / normalize','Preprocessing должен соответствовать encoder и его весам.','input'),
            V('text',2,0,'Текст','tokenize + special tokens','Токенизация задаётся text encoder. Это не обязательная BERT-последовательность.','input'),
            V('vision',0,1,'Image encoder','ViT / ConvNet','Глобальные visual features затем проецируются в общее пространство.'),
            V('language',2,1,'Text encoder','вектор текста','В исходном CLIP используется causal text transformer; конкретный SigLIP text tower определяется его архитектурой.'),
            V('iproj',0,2,'Projection + L2 norm','image embedding u','Проекция согласует ширину. Нормированный dot product становится cosine similarity.'),
            V('tproj',2,2,'Projection + L2 norm','text embedding v','Текстовые embeddings можно сохранить для retrieval / zero-shot кандидатов.'),
            V('scores',1,3,'Матрица image–text','N×N scores','В CLIP logits равны t·cosine; SigLIP использует t·cosine+b. При diagonal-only разметке N positives и N²−N negatives.'),
            V('loss',1,4,loss,'обучение по парам','CLIP нормирует softmax по строкам и столбцам. SigLIP суммирует signed logistic terms; scale и bias обучаются.','loss')],
          [('image','vision'),('text','language'),('vision','iproj'),('language','tproj'),('iproj','scores'),('tproj','scores'),('scores','loss','train','training')])

    D('coca','coca','CoCa: contrastive + captioning',
      'Нижний unimodal decoder сохраняет независимость text embedding. Верхний multimodal decoder читает image memory через cross-attention.',[
        V('image',0,0,'Изображение','visual input','Изображение кодируется независимо от подаваемого текста.','input'),
        V('tokens',2,0,'Text tokens','causal prefix','На train caption-последовательность известна, на generation продолжение выбирается по шагам.','input'),
        V('vision',0,1,'Image encoder','visual tokens V','Набор визуальных токенов сохраняется для cross-attention.'),
        V('uni',2,1,'Unimodal decoder','без image cross-attn','Нижние causal layers читают только текстовый контекст.'),
        V('ipool',0,2,'Image pooling','глобальный embedding','В исходном подходе используются обучаемые attention poolers для согласованных visual representations.'),
        V('tpool',2,2,'Text pooling / CLS','глобальный embedding','Финальная агрегирующая текстовая позиция unimodal пути задаёт embedding для contrastive loss.'),
        V('multi',2,3,'Multimodal decoder','text queries + visual K,V','Верхняя часть читает unimodal hidden states и visual tokens; не только один pooled image vector.'),
        V('contrastive',0,4,'Contrastive loss','image ↔ text','Сопоставляет независимые глобальные представления.','loss'),
        V('caption',2,4,'Vocabulary → caption loss','следующие токены','Условный token loss по изображению и текстовому префиксу. Общая цель — взвешенная сумма двух loss.','loss')],
      [('image','vision'),('tokens','uni'),('vision','ipool'),('uni','tpool'),('uni','multi','hidden states','context'),('vision','multi','visual K,V','context'),('ipool','contrastive'),('tpool','contrastive'),('multi','caption','train','training')])

    C('navit','navit','NaViT: packing изображений',
      'Разные изображения упаковываются в общий поток, но остаются независимыми примерами. Внутри каждой картинки attention двунаправленный.',[
        ('Изображения разных форм','Hᵢ×Wᵢ → Nᵢ patches','Разные aspect ratios и длины сохраняют spatial-структуру в рамках выбранного resize/budget.','input'),
        ('Patch embedding + позиции','координаты row / column','Позиции относятся к собственной сетке изображения, а не к произвольному offset в packed-буфере.'),
        ('Packing + image IDs','общий буфер токенов','Принадлежность каждому изображению сохраняется вместе с valid/padding информацией.'),
        ('Masked Transformer','полные диагональные блоки','Attention запрещает межкартинные связи. Плотный kernel при этом не обязан физически пропускать запрещённые пары.'),
        ('Pooling по изображениям','отдельный embedding каждому','Общий pooling всего пакета снова смешал бы независимые примеры, поэтому агрегация согласована с image IDs.'),
        ('Голова / objective','targets своих изображений','Результаты сопоставляются с исходными примерами, а не с порядком случайных соседей в пакете.','output')])
    C('naflex','naflex','SigLIP2 / NaFlex: гибкий visual input',
      'Показан visual путь с гибкой сеткой и бюджетом патчей. NaFlex — способ обработки visual input, а не отдельный caption decoder.',[
        ('Исходное изображение','произвольные пропорции','Размеры исходного изображения задают желательное отношение сторон.','input'),
        ('Подбор размера сетки','h×w ≤ max_tokens','Processor выбирает допустимый размер с учётом кратности patch size. Округление может приближать точные пропорции.'),
        ('Resize → patch embedding','реальные токены + padding','Реальная длина и маска сохраняются. Максимум токенов — предел, не обязательная фактическая длина.'),
        ('Position + vision encoder','учесть форму h×w','Позиционные представления должны соответствовать реальной прямоугольной сетке.'),
        ('Masked pooling + projection','image embedding','Padding исключается из полезной агрегации, затем признаки согласуются с текстовым пространством.'),
        ('Image–text matching','SigLIP-style objective','Text tower остаётся независимым путём; обучение соответствует objective выбранной SigLIP2 модели.','output')])

    D('moe','moe','MoE: router и выбранные FFN',
      'Показана замена FFN на top-k экспертов. Attention и другие общие компоненты transformer-блока остаются общими.',[
        V('x',1,0,'Токен x','[d]','Разные токены одного изображения могут выбирать разных экспертов.','input'),
        V('router',1,1,'Router → top-k','scores E экспертов','Из scores выбираются k экспертов; routing weights нормируются согласно выбранному соглашению.'),
        V('e0',0,2,'Expert A: FFN','выбранный эксперт','Один из выбранных независимых FFN. Все экспертные веса существуют в системе, даже если этот токен их не использует.'),
        V('e1',2,2,'Expert B: FFN','пример top-2','Второй выбранный FFN. На рисунке k=2 только для иллюстрации; это не фиксированный k всех MoE.'),
        V('sum',1,3,'Взвешенная сумма','Σ gateᵢ · Expertᵢ(x)','Суммируются выходы выбранных экспертов. Гейты не равнозначны attention weights по spatial-токенам.'),
        V('out',1,4,'Выход MoE FFN','в общий residual block','Далее используются residual/norm-компоненты конкретного transformer-блока.','output')],
      [('x','router'),('router','e0','gate A'),('router','e1','gate B'),('e0','sum'),('e1','sum'),('sum','out')])

    D('rope','rope','RoPE внутри attention',
      'RoPE вращает Q и K до вычисления scores. Values обычно проходят обычную проекцию без этого вращения.',[
        V('x',1,0,'Hidden states X','token representations','Общие содержательные представления перед проекциями.','input'),
        V('q',0,1,'Q projection','queries','Проекция XWQ задаёт содержательные queries.'),V('k',1,1,'K projection','keys','Проекция XWK задаёт содержательные keys.'),V('v',2,1,'V projection','values','Values несут смешиваемое содержимое.'),
        V('rq',0,2,'RoPE(Q, positions)','попарные вращения','Для query используется его позиция и согласованное разбиение координат.'),V('rk',1,2,'RoPE(K, positions)','то же соглашение частот','Фиксированные q,k взаимодействуют через относительное вращение. Содержание в score сохраняется.'),
        V('scores',0,3,'Scores → mask → softmax','по ключам','Вращённые queries/keys задают позиционно-зависимые scores.'),V('out',1,4,'Взвесить V','attention output','RoPE не заменяет агрегацию values и не является loss.','output')],
      [('x','q'),('x','k'),('x','v'),('q','rq'),('k','rk'),('rq','scores'),('rk','scores'),('scores','out','weights'),('v','out','V','context')])

    D('kv-cache','cache','KV-cache: один causal decode step',
      'Схема одного слоя и одной новой позиции. Cache хранит прошлые K/V того же префикса и слоя, а не заранее известное продолжение.',[
        V('new',0,0,'Новый hidden state','одна позиция','Текущий токен проходит проекции и согласованное позиционное кодирование.','input'),
        V('old',2,0,'Прошлый KV-cache','слой l; тот же префикс','Старые состояния не зависят от будущего токена благодаря causal-доступу.','input'),
        V('proj',0,1,'Qnew, Knew, Vnew','по текущему токену','Прошлые Q для нового attention-выхода не нужны.'),
        V('append',2,2,'Дополнить K/V','прошлое + новая позиция','Cache растёт по контексту. В эффективной реализации используется подготовленный буфер.'),
        V('attn',0,3,'Qnew читает весь KV','стоимость растёт с T','Cache убирает повторные проекции прошлого, но новый query всё ещё сравнивается с растущей историей.'),
        V('out',0,4,'Выход нового токена','следующий подслой','Эта операция повторяется по слоям decoder; vocabulary head находится после их стека.','output')],
      [('new','proj'),('old','append'),('proj','append','Knew,Vnew','context'),('proj','attn','Qnew'),('append','attn','K,V','context'),('attn','out')])

    add_cnn(D,C,V)
    add_detectors(D,C,V)
    for t in topics:t['diagrams']=[d['id'] for d in diagrams if d['topic']==t['id']]
    return diagrams


def add_cnn(D,C,V):
    C('vgg','resnet','VGG: последовательные conv-блоки',
      'Схема семейства VGG: одинаковые маленькие ядра складываются в глубину. Число conv-слоёв и размеры head зависят от варианта VGG.',[
        ('Изображение','[B,3,H,W]','Вход CNN в согласованном размере и с preprocessing весов.','input'),
        ('Conv 3×3 → ReLU','несколько раз','Маленькие spatial-ядра последовательно увеличивают effective receptive field.'),
        ('MaxPool → следующий блок','меньше H,W; больше C','Группы conv-операций и pooling повторяются. В базовой VGG нет residual shortcut.'),
        ('Flatten → FC layers','dense classifier','В классической конфигурации после spatial-backbone стоят полносвязные слои.'),
        ('Class logits','голова классификации','Выходы соответствуют классам обучающей задачи.','output')])
    C('resnet','resnet','ResNet: backbone из residual stages',
      'Схема сети целиком. У отдельных вариантов разные stem, basic/bottleneck blocks и число повторений.',[
        ('Изображение → stem','conv / norm / activation','Stem начинает извлечение признаков; конкретные kernel и stride задаёт вариант сети.','input'),
        ('Residual stage 1','несколько residual blocks','Внутри совместимых форм shortcut может быть identity.'),
        ('Residual stages 2…4','downsample + рост C','На переходах между формами shortcut согласуется проекцией/stride.'),
        ('Global average pooling','[B,C]','Spatial-оси агрегируются в один вектор на изображение.'),
        ('Linear classifier','class logits','Backbone может также передавать intermediate feature maps detector-голове.','output')])
    D('residual-block','resnet','Residual block: основной и обходной пути',
      'Пример basic block при одинаковых входной и выходной формах. Для изменения формы нужен projection shortcut.',[
        V('x',1,0,'Вход x','[B,C,H,W]','Одинаковая форма обязательна для показанного identity shortcut.','input'),
        V('conv1',1,1,'Conv → BN → ReLU','первая свёртка','Основная ветвь начинает обучаемое преобразование F(x).'),
        V('conv2',1,2,'Conv → BN','вторая свёртка','Формирует residual-признаки той же формы.'),
        V('add',1,3,'Сумма + ReLU','x + F(x)','К основному пути прибавляется shortcut. В bottleneck блоке число conv-подслоёв другое.'),
        V('out',1,4,'Выход блока','следующий residual block','Сумма не увеличивает число каналов, в отличие от DenseNet concatenation.','output')],
      [('x','conv1'),('conv1','conv2'),('conv2','add'),('x','add','identity','skip'),('add','out')])
    D('densenet','resnet','DenseNet: конкатенация признаков',
      'Внутри dense block слой читает результаты предшествующих слоёв. Между dense blocks стоят transition-операции.',[
        V('x0',0,0,'Stem / вход блока','x₀','Начальные карты признаков.','input'),
        V('h1',1,1,'Dense layer H₁','x₁ = H₁(x₀)','Выдаёт новые feature channels.'),
        V('cat',1,2,'Concatenate','[x₀, x₁]','Каналы объединяются, а не складываются попарно.'),
        V('h2',1,3,'Dense layer H₂','x₂ = H₂([x₀,x₁])','Каждый следующий слой получает накопленные признаки.'),
        V('transition',1,4,'Concat → transition','compression / downsample','Transition связывает dense blocks, уменьшая spatial-размер и при необходимости число каналов.'),
        V('head',1,5,'Дальше blocks → GAP → head','class logits','Схема показывает принцип dense connectivity, а не фиксированное число слоёв.','output')],
      [('x0','h1'),('h1','cat'),('x0','cat','старые каналы','context'),('cat','h2'),('h2','transition'),('cat','transition','сохранить признаки','context'),('transition','head')])
    C('depthwise','conv','Depthwise separable convolution',
      'Spatial-смешивание и смешивание каналов разделены. Это факторизация операции, используемая в мобильных CNN.',[
        ('Вход','Cin каналов','Признаки изображения или предыдущего слоя.','input'),
        ('Depthwise k×k','по одному фильтру на канал','При multiplier=1 каждый входной канал обрабатывается отдельно: k²Cin весов.'),
        ('Pointwise 1×1','Cin → Cout','Смешивает каналы: Cin·Cout весов без bias.'),
        ('Выход','Cout каналов','Norm/activation могут стоять между операциями согласно архитектуре; здесь показана линейная факторизация.','output')])
    C('mobilenet-v1','mobile','MobileNetV1: depthwise + pointwise blocks',
      'Принцип MobileNetV1: обычный stem и повторяющиеся depthwise separable blocks вместо дорогих полных spatial-conv.',[
        ('Stem Conv','начальные признаки','Обычная свёртка начинает backbone.','input'),
        ('Depthwise → BN → ReLU','spatial kernel','Каждый канал фильтруется независимо.'),
        ('Pointwise → BN → ReLU','смешать каналы','1×1-проекция задаёт следующую ширину признаков.'),
        ('Повторить blocks','часть blocks со stride 2','Постепенно уменьшается resolution и меняется ширина сети.'),
        ('Global pooling → classifier','image logits','Depthwise отдельно не является всей MobileNet: необходимы channel mixing и организация stages.','output')])
    D('mobilenet-v2','mobile','MobileNetV2: inverted residual block',
      'Схема inverted bottleneck: сначала расширить каналы, затем spatial-conv, затем линейно сжать. Shortcut допустим при stride=1 и одинаковой ширине.',[
        V('x',1,0,'Узкий вход','Cin каналов','Начальная низкая ширина bottleneck.','input'),
        V('expand',1,1,'1×1 expand + activation','больше каналов','Расширяет пространство для нелинейного преобразования.'),
        V('dw',1,2,'Depthwise + activation','spatial filtering','Stride может уменьшить resolution.'),
        V('project',1,3,'1×1 linear projection','сжать до Cout','Линейный bottleneck не ставит обычную ReLU после этой проекции.'),
        V('add',1,4,'Shortcut при совместимости','stride=1, Cin=Cout','Иначе выход берётся без identity-суммы.','output')],
      [('x','expand'),('expand','dw'),('dw','project'),('project','add'),('x','add','если формы равны','skip')])
    D('se','mobile','Squeeze-and-Excitation: gates каналов',
      'SE строит глобальные gates для каждого канала и умножает ими исходную spatial-карту.',[
        V('x',0,0,'Feature map X','[B,C,H,W]','Признаки, которые нужно перевзвесить.','input'),
        V('pool',2,1,'Global average pooling','[B,C]','Squeeze агрегирует spatial-оси отдельно по каналам.'),
        V('mlp',2,2,'FC → activation → FC','C → C/r → C','Excitation моделирует взаимодействие каналов.'),
        V('gate',2,3,'Sigmoid gates','[B,C,1,1]','Каналы имеют отдельные gates; это не softmax с суммой 1.'),
        V('mul',0,4,'Умножение X · gates','broadcast по H,W','Spatial-размер сохраняется, изменяется вклад каждого канала.','output')],
      [('x','pool'),('pool','mlp'),('mlp','gate'),('gate','mul','gates','context'),('x','mul','исходные признаки','context')])
    C('efficientnet','mobile','EfficientNet: MBConv + SE stages',
      'Типичная структура семейства EfficientNet из лекции. Compound scaling согласует глубину, ширину и разрешение; это не дополнительный слой после head.',[
        ('Изображение → stem','начальные карты','Input resolution входит в выбранную конфигурацию.','input'),
        ('MBConv + SE stages','expand → DW → SE → project','Повторяющиеся inverted bottleneck blocks; у stages свои widths, repeats и strides. Линейная проекция и допустимый shortcut входят в каждый такой блок.'),
        ('Head conv','итоговая ширина каналов','Pointwise head подготавливает признаки последней стадии.'),
        ('Global average pooling','глобальные признаки','Агрегирует spatial-оси перед финальным classifier.'),
        ('Dropout → classifier','class logits','Depth/width/resolution масштабируются согласованно для выбранного варианта.','output')])
    D('resnext','moderncnn','ResNeXt: grouped bottleneck',
      'Cardinality — число групп преобразований. На схеме показана grouped-conv реализация residual bottleneck.',[
        V('x',1,0,'Вход','residual shortcut','Входная форма должна совпасть с выходом либо используется projection.','input'),
        V('reduce',1,1,'1×1 bottleneck','согласовать ширину','Pointwise-свёртка задаёт число промежуточных каналов.'),
        V('group',1,2,'Grouped spatial conv','G независимых групп','При G группах spatial-conv использует в G раз меньше весов относительно полной conv тех же размеров.'),
        V('expand',1,3,'1×1 projection','выходная ширина','Каналы снова смешиваются pointwise-операцией.'),
        V('add',1,4,'Add shortcut + activation','residual output','Группы не являются отдельными классами классификации.','output')],
      [('x','reduce'),('reduce','group'),('group','expand'),('expand','add'),('x','add','shortcut','skip')])
    D('resnest','moderncnn','ResNeSt: split-attention block',
      'Gating-ветвь вычисляет веса, которые затем применяются к исходным split-признакам.',[
        V('x',0,0,'Вход → bottleneck','residual CNN features','Входные признаки подготавливаются к преобразованиям.','input'),
        V('split',0,1,'Split transformations','несколько feature-ветвей','Строятся согласованные группы и splits признаков.'),
        V('pool',2,2,'Aggregate → global pool','общий контекст','Агрегированные признаки используются для выбора весов.'),
        V('gates',2,3,'Gating по splits','attention weights','Веса нормируются по splits согласно split-attention блоку.'),
        V('sum',0,4,'Weighted feature sum','признаки × gates','Исходные feature-ветви, а не только pooled-вектор, складываются с вычисленными весами.'),
        V('out',0,5,'Projection + residual','выход блока','Завершается residual bottleneck.','output')],
      [('x','split'),('split','pool'),('pool','gates'),('split','sum','features','context'),('gates','sum','gates','context'),('sum','out')])
    D('convnext','moderncnn','ConvNeXt: modern convolutional block',
      'Один ConvNeXt-подобный блок: depthwise spatial-mixing, LayerNorm и channel MLP. Полная сеть дополнительно содержит stem и downsampling между stages.',[
        V('x',1,0,'Вход X','NCHW feature map','Residual проходит в обход основной ветви при совпадающих формах.','input'),
        V('dw',1,1,'Depthwise 7×7','spatial mixing','Большое depthwise-ядро смешивает соседние spatial-позиции отдельно по каналам.'),
        V('ln',1,2,'LayerNorm по каналам','перестановка NCHW ↔ NHWC','Нужно согласовать ось LN; просто LayerNorm(C) на NCHW обычно нормирует другую ось.'),
        V('mlp',1,3,'Pointwise MLP','C → 4C → C; GELU','Обучаемое смешивание каналов в каждой spatial-позиции.'),
        V('scale',1,4,'Layer scale / DropPath','зависит от варианта','Полезные компоненты конфигурации блока; DropPath влияет на residual-ветвь в train.'),
        V('add',1,5,'Residual sum','X + branch(X)','Возвращает карту той же формы для следующего блока.','output')],
      [('x','dw'),('dw','ln'),('ln','mlp'),('mlp','scale'),('scale','add'),('x','add','identity','skip')])


def add_detectors(D,C,V):
    for key,title,steps in [
      ('r-cnn','R-CNN: CNN по отдельным proposals',[
        ('Изображение → proposals','Selective Search','Внешний алгоритм строит region proposals.','input'),
        ('Crop / resize каждого ROI','отдельные изображения регионов','Каждый proposal преобразуется к входу CNN.'),
        ('CNN для каждого crop','повторные вычисления','Backbone выполняется отдельно для множества перекрывающихся regions.'),
        ('SVM + box regression','отдельные предсказания ROI','Классическая R-CNN использует классификаторы и регрессию по извлечённым признакам.'),
        ('Фильтрация → NMS','итоговые detections','Постобработка уменьшает число дублирующих boxes.','output')]),
      ('fast-r-cnn','Fast R-CNN: общая feature map',[
        ('Изображение → CNN','одна общая карта','Backbone считается один раз для изображения.','input'),
        ('Внешние proposals','Selective Search','Предложения всё ещё строятся отдельным алгоритмом.'),
        ('ROI Pooling по feature map','fixed-size ROI features','Области из списка proposals индексируют общую карту, а не запускают CNN с нуля.'),
        ('Shared head → class + box','две обучающие цели','Classification и box regression обучаются по ROI-признакам.'),
        ('Post-processing','scores, boxes, NMS','Получаем итоговые detections.','output')]),
      ('faster-r-cnn','Faster R-CNN: backbone + RPN + ROI head',[
        ('Изображение → backbone','shared feature map','Общая CNN-карта используется RPN и ROI-головой.','input'),
        ('RPN по anchors','objectness + offsets','Region Proposal Network предсказывает кандидатные regions из общей карты.'),
        ('Decode / filter proposals','proposal NMS','Отбираются regions для второй стадии. Общая feature map остаётся доступной ROI-операции.'),
        ('ROI features → final head','classification + box refine','ROI Pooling исторического Faster R-CNN либо другой ROI-оператор конкретной реализации.'),
        ('Score filtering → NMS','итоговые detections','Objectness RPN и финальные class scores решают разные задачи.','output')])]:
        if key=='fast-r-cnn':
            D(key,'rcnn',title,'Backbone и внешний proposal-алгоритм читают одно изображение; ROI Pooling объединяет карту и regions.',[
                V('image',1,0,'Изображение','общий вход','Оба пути получают одну сцену.','input'),
                V('cnn',0,1,'CNN backbone','shared feature map','Вычисляется один раз для изображения.'),
                V('proposals',2,1,'Selective Search','внешние proposals','В отличие от Faster R-CNN, proposals ещё не выдаются RPN.'),
                V('roi',1,2,'ROI Pooling','fixed-size ROI features','Использует регионы и общую feature map.'),
                V('heads',1,3,'Class + box heads','две цели по ROI','Классификация и регрессия обучаются по region features.'),
                V('out',1,4,'Filtering → NMS','detections','Финальная постобработка.','output')],
              [('image','cnn'),('image','proposals'),('cnn','roi','feature map','context'),('proposals','roi','regions'),('roi','heads'),('heads','out')])
        elif key=='faster-r-cnn':
            D(key,'rcnn',title,'RPN строит proposals, а ROI-голова повторно использует ту же общую feature map.',[
                V('cnn',0,0,'Изображение → backbone','shared feature map','Одна CNN-карта для RPN и второй стадии.','input'),
                V('rpn',2,1,'RPN по anchors','objectness + offsets','Кандидаты regions формируются обучаемой сетью.'),
                V('proposals',2,2,'Decode + proposal NMS','region proposals','Первичный отбор regions до финального ROI classifier.'),
                V('roi',0,3,'ROI feature extraction','map + regions','ROI Pooling исторического подхода или другой согласованный ROI-оператор реализации.'),
                V('head',0,4,'Final class + box head','refine proposals','Objectness RPN не равен финальному class score.'),
                V('out',0,5,'Score filter + NMS','detections','Постобработка второй стадии.','output')],
              [('cnn','rpn'),('rpn','proposals'),('cnn','roi','shared map','context'),('proposals','roi','regions'),('roi','head'),('head','out')])
        else:
            C(key,'rcnn',title,'Схема исторического pipeline из лекции; конкретные современные реализации могут использовать другие heads.',steps)
    C('ssd','ssd','SSD: multiscale default boxes',
      'One-stage detector предсказывает классы и offsets по нескольким feature levels. Hard negative mining относится к обучению, а не к дополнительной inference-голове.',[
        ('Изображение → backbone','основные feature maps','CNN извлекает пространственные признаки.','input'),
        ('Дополнительные feature levels','несколько масштабов','Разные уровни имеют различные resolution и receptive fields.'),
        ('Default boxes на уровнях','размеры / aspect ratios','На позициях заранее заданы опорные boxes.'),
        ('Dense class + box heads','предсказания по default box','Все кандидаты выдаются без отдельной ROI-стадии.'),
        ('Decode offsets → NMS','boxes, classes, scores','На train выполняются matching и hard negative mining с ограничением числа negatives.','output')])
    C('yolov1','yolo','YOLOv1: grid prediction',
      'Историческая YOLOv1; её формат выхода нельзя автоматически переносить на YOLOv3 или современные YOLO.',[
        ('Изображение → CNN','глобальный feature extractor','Одно изображение обрабатывается общей сетью.','input'),
        ('Grid S×S','назначение по центру объекта','Ячейка отвечает за объект, центр которого в ней расположен.'),
        ('B boxes + C classes / cell','S×S×(5B+C)','В YOLOv1 class probabilities общие для boxes одной ячейки; это не формат B×(5+C).'),
        ('Decode boxes + confidence','оценки кандидатов','Confidence и conditional class probabilities используются согласно постановке YOLOv1.'),
        ('Фильтрация + NMS','detections','Дубликаты и слабые кандидаты отбрасываются.','output')])
    C('yolov3','yolo','YOLOv3: anchors и несколько scales',
      'Схема YOLOv3 из лекции: anchor-based dense prediction на нескольких разрешениях.',[
        ('Изображение → Darknet','backbone с residual blocks','Общая CNN формирует карты нескольких масштабов.','input'),
        ('Multi-scale feature paths','upsample + feature fusion','Информация с разных уровней используется prediction heads.'),
        ('Anchor-based heads','offsets / objectness / classes','Head предсказывает параметры для anchors каждой spatial-ячейки.'),
        ('Decode всех уровней','к общей системе координат','Grid offsets и anchor scales преобразуются в boxes изображения.'),
        ('Scores → per-class NMS','detections','Эта архитектура отличается от grid-output YOLOv1.','output')])
    D('fpn','retinanet','Feature Pyramid Network (FPN)',
      'Top-down и lateral пути собирают карты согласованной ширины. Показаны три уровня для ясности; число уровней задаёт модель.',[
        V('c3',0,0,'C3 backbone','fine resolution','Более подробная spatial-карта.','input'),V('c4',1,0,'C4 backbone','medium resolution','Промежуточный масштаб.','input'),V('c5',2,0,'C5 backbone','coarse resolution','Более грубые semantic features.','input'),
        V('m3',0,2,'1×1 lateral + top-down','M3','К lateral C3 добавляют upsample M4 с согласованными spatial-размерами.'),V('m4',1,2,'1×1 lateral + top-down','M4','К lateral C4 добавляют upsample M5.'),V('m5',2,2,'1×1 lateral','M5','Верхний уровень начинает top-down поток.'),
        V('p3',0,3,'3×3 → P3','fine output','Сглаживающая свёртка формирует выходной feature level.','output'),V('p4',1,3,'3×3 → P4','medium output','Второй выход пирамиды.','output'),V('p5',2,3,'3×3 → P5','coarse output','Третий выход пирамиды. Дополнительные levels зависят от detector.','output')],
      [('c3','m3'),('c4','m4'),('c5','m5'),('m5','m4','upsample','context'),('m4','m3','upsample','context'),('m3','p3'),('m4','p4'),('m5','p5')])
    C('retinanet','retinanet','RetinaNet: FPN и две dense heads',
      'FPN отвечает за multiscale признаки; focal loss — за обучение dense classification при большом числе фоновых anchors.',[
        ('Изображение → backbone','CNN features','Общая CNN формирует набор уровней.','input'),
        ('FPN','P3, P4, …','Пирамида строит несколько resolution с согласованной шириной.'),
        ('Anchors + две heads','classification / regression','На каждом уровне одинаковая логика dense heads выдаёт scores классов и offsets boxes.'),
        ('Decode / score filtering','кандидаты всех уровней','На train classification обучается focal loss, regression — по назначенным anchors.'),
        ('Per-class NMS','detections','NMS — постобработка, focal loss — обучающий objective.','output')])
    D('centernet','centernet','CenterNet: Objects as Points',
      'Три выхода по общей карте: heatmap центров, размеры и offsets. Речь об Objects as Points, а не о другой статье CenterNet: Keypoint Triplets.',[
        V('image',1,0,'Изображение','input','Общая сцена, в которой нужно найти объекты.','input'),V('backbone',1,1,'Backbone / upsampling','feature map со stride s','Сеть формирует spatial-карту для dense предсказаний.'),
        V('heat',0,2,'Center heatmap','по классам','Ищет положения центров объектов.'),V('size',1,2,'Width / height head','размеры объектов','Размеры восстанавливают bbox вокруг центра.'),V('offset',2,2,'Offset head','дробная поправка центра','Компенсирует квантование центра на coarse feature grid.'),
        V('decode',1,3,'Peaks + размеры + offset','центр → bbox','Local-max поиск на heatmap, выбор кандидатов и согласованный decode дают boxes.','output')],
      [('image','backbone'),('backbone','heat'),('backbone','size'),('backbone','offset'),('heat','decode'),('size','decode'),('offset','decode')])
    C('cornernet','centernet','CornerNet: пары углов',
      'Схема для сравнения с CenterNet: объект параметризуется top-left и bottom-right углами.',[
        ('Изображение → backbone','общая feature map','CNN извлекает признаки сцены.','input'),
        ('Угловые feature paths','corner pooling','Признаки подготавливаются для двух типов углов.'),
        ('Две heatmaps + offsets','top-left / bottom-right','Углы локализуются на feature grid с поправками квантования.'),
        ('Grouping embeddings','какие углы одной рамки','Дополнительные признаки помогают сопоставить top-left и bottom-right одного объекта.'),
        ('Pair corners → boxes','detections','Найти углы и правильно объединить их — разные задачи; центр не является единственным ключевым выходом.','output')])
    D('fcos','fcos','FCOS: class, ltrb и centerness',
      'Anchor-free dense detector: каждая подходящая location feature map задаёт расстояния до сторон bbox.',[
        V('input',1,0,'Изображение → backbone','spatial features','Общий extractor сцены.','input'),V('fpn',1,1,'FPN','несколько feature levels','Масштабы распределяют объекты по уровням.'),
        V('cls',0,2,'Classification branch','class scores','Dense classification обучается focal loss.'),V('reg',1,2,'Box regression','l, t, r, b','Расстояния относятся к текущей location в согласованных единицах.'),V('center',2,2,'Centerness branch','предсказанный score','На train target вычисляется по GT-расстояниям. На inference centerness предсказывается, GT неизвестны.'),
        V('decode',1,3,'Decode + score fusion','(x−l,y−t,x+r,y+b)','Boxes и scores объединяются из трёх ветвей.'),V('nms',1,4,'Filtering + NMS','detections','Anchor-free не означает отказ от обычной постобработки FCOS.','output')],
      [('input','fpn'),('fpn','cls'),('fpn','reg'),('fpn','center'),('cls','decode'),('reg','decode'),('center','decode'),('decode','nms')])
    D('detr','detr','DETR: encoder–decoder set prediction',
      'Object queries — обучаемые слоты множества. Они не являются causal-последовательностью объектов. Hungarian matching используется только для обучения.',[
        V('image',0,0,'Изображение → CNN','feature map','Backbone создаёт карту признаков.','input'),V('flatten',0,1,'Projection + flatten + PE','image tokens','Каналы согласуются с d, spatial-позиции сохраняются кодированием.'),V('encoder',0,2,'Transformer encoder','image memory','Полное self-attention создаёт контекстную visual memory.'),
        V('queries',2,2,'Learned object queries','Q слотов','Инициализируют слоты decoder; конкретный GT не задан им навсегда.','input'),
        V('decoder',1,3,'Transformer decoder','query self-attn + cross-attn','Queries взаимодействуют друг с другом и читают image memory. Вычисление слотов параллельно.'),
        V('heads',1,4,'Class / no-object + bbox','предсказания всех Q слотов','Общие heads применяются к каждому decoder output.'),V('infer',0,5,'Inference detections','фильтр no-object / scores','В стандартном DETR нет обычного обязательного bbox NMS; matching с GT на inference не выполняется.','output'),
        V('gt',2,4,'GT: классы и boxes','только обучение','Известная разметка нужна для назначения и loss.','input'),V('matching',2,5,'Hungarian → set loss','one-to-one assignment','Назначенные slots получают class/box loss; unmatched slots учатся no-object.','loss')],
      [('image','flatten'),('flatten','encoder'),('encoder','decoder','memory K,V','context'),('queries','decoder','queries'),('decoder','heads'),('heads','infer'),('heads','matching','predictions','training'),('gt','matching','GT','training')])
    C('hog-svm','classicaldetectors','HOG + linear SVM',
      'Классический detector по скользящим окнам и пирамиде масштаба; это другой pipeline, чем Viola–Jones.',[
        ('Image pyramid → windows','несколько размеров объектов','Скользящее окно применяется на нескольких масштабах.','input'),
        ('Gradients → HOG','гистограммы ориентаций','Локальные градиенты агрегируются в ячейки и нормированные блоки.'),
        ('Linear SVM score','wᵀφ(window)+b','Классификатор работает по descriptor выбранного окна.'),
        ('Threshold / grouping','окна объектов','Соседние детекции объединяются или подавляются согласно реализации.','output')])
    C('viola-jones','classicaldetectors','Viola–Jones: cascade detector',
      'Integral image ускоряет прямоугольные Haar-признаки, а каскад отбрасывает большую часть окон на ранних стадиях.',[
        ('Изображение → integral image','быстрые суммы областей','Прямоугольные суммы считаются по нескольким обращениям к prefix sums.','input'),
        ('Sliding windows / scales','кандидатные регионы','Объекты разных размеров проверяются на разных масштабах.'),
        ('Haar features → cascade','ранний отказ фоновых окон','Стадии классификатора последовательно фильтруют кандидатов; прошедшее окно идёт дальше.'),
        ('Положительные окна → grouping','итоговые detections','Согласуются соседние срабатывания.','output')])
