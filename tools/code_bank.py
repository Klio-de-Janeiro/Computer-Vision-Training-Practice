"""Small independently authored API exercises. Fixtures stay separate from the operation."""
from copy import deepcopy
from textwrap import dedent


def build(topics, questions):
    cards=[]; lookup={t['id']:t for t in topics}
    def C(key,lib,related,focus,prompt,setup,code,result,why,trap,test='',doc='',runtime='cpu'):
        ids=related.split();topic=lookup[ids[0]]
        card=dict(id=key,library=lib,topics=ids,lesson=topic['lesson'],focus=focus,prompt=prompt,
          setup=dedent(setup).strip(),code=dedent(code).strip(),result=result,explanation=why,trap=trap,
          test=dedent(test).strip(),runtime=runtime,source=deepcopy(topic['source']),
          basis='Дополнительная практика API по теме курса; учебные данные, не готовое решение домашней работы.',
          doc=doc or ('https://numpy.org/doc/stable/reference/'+('generated/'+focus+'.html' if focus.startswith('numpy.') else 'index.html') if lib=='NumPy' else 'https://docs.pytorch.org/docs/2.14/'+('generated/'+focus+'.html' if focus.startswith('torch.') else 'index.html')))
        if not doc:
            if focus.startswith('Tensor.'):
                card['doc']='https://docs.pytorch.org/docs/2.14/generated/torch.'+focus+'.html'
            elif focus.startswith('torch.nn.Module.'):
                card['doc']='https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#'+focus
            elif focus.startswith('Optimizer.'):
                card['doc']='https://docs.pytorch.org/docs/2.14/generated/torch.optim.'+focus+'.html'
            elif focus.startswith('ndarray.'):
                card['doc']='https://numpy.org/doc/stable/reference/generated/numpy.'+focus+'.html'
            elif focus=='Generator.choice':
                card['doc']='https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html'
            elif focus.startswith('torch.utils.data.'):
                card['doc']='https://docs.pytorch.org/docs/2.14/data.html#'+focus
            elif focus=='basic slicing':
                card['doc']='https://numpy.org/doc/stable/user/basics.indexing.html'
        cards.append(card)
        questions.append(dict(id='api-'+key,lesson=topic['lesson'],topic=topic['id'],collection='code',type='open',level='apply',
          prompt=focus+': '+prompt+'\n\nДано:\n```python\n'+card['setup']+'\n```',
          answer='```python\n'+card['code']+'\n```\n\n'+why+'\n\nОжидаемый результат: '+result,
          example=trap,rubric=['Правильно выбрана операция и аргументы','Обоснованы форма, dtype или результат','Объяснена указанная тонкость'],source=deepcopy(topic['source']),codeCard=key))

    C('np-array','NumPy','raster','numpy.array','Создайте RGB-пиксель с dtype float32 из values.',
      'values = [255, 128, 0]','out = np.array(values, dtype=np.float32)','[255.,128.,0.], shape=(3,).','Явный dtype позволяет дальнейшую дробную арифметику. Создание массива само не нормирует диапазон.','astype/array float не делит на 255.', 'assert out.dtype == np.float32 and out.tolist()==values')
    C('np-arange','NumPy','geometry','numpy.arange','Создайте горизонтальные координаты 0…4 без включения 5.',
      'width = 5','out = np.arange(width)','[0,1,2,3,4].','arange использует исключённую правую границу. Координаты индексов здесь целые.','Для дробной равномерной сетки с включённым концом полезнее linspace.','assert out.tolist()==[0,1,2,3,4]')
    C('np-linspace','NumPy','brightness','numpy.linspace','Постройте 5 уровней от 0 до 1 включительно.',
      'count = 5','out = np.linspace(0, 1, count)','[0,.25,.5,.75,1].','linspace задаёт число точек, а не шаг.','arange(0,1,.25) не включит 1.','np.testing.assert_allclose(out,[0,.25,.5,.75,1])')
    C('np-zeros','NumPy','segmentation-task','numpy.zeros','Создайте пустую булеву маску 2×3.',
      'shape = (2, 3)','out = np.zeros(shape, dtype=bool)','Шесть False, shape=(2,3).','Булева маска хранит принадлежность foreground.','uint8 с ID классов и bool foreground имеют разный смысл.','assert out.shape==(2,3) and not out.any()')
    C('np-clip','NumPy','brightness image-pipeline','numpy.clip','Ограничьте интенсивности 0…255 после вычисления во float.',
      'x = np.array([-20., 80., 270.])','out = np.clip(x, 0, 255)','[0.,80.,255.].','Clip насыщает значения на концах, а не зацикливает unsigned-арифметику.','Clip после уже случившегося переполнения не восстановит потерянные значения.','assert out.tolist()==[0,80,255]')
    C('np-astype','NumPy','raster image-pipeline','ndarray.astype','Подготовьте uint8 массив для безопасного вычитания.',
      'x = np.array([0, 255], dtype=np.uint8)','out = x.astype(np.float32)','float32 [0.,255.].','Перевод выполняется до разности и возведения в квадрат MSE.','Вычесть uint8, а потом перевести в float — поздно.','assert out.dtype==np.float32 and (out-255).tolist()==[-255,0]')
    C('np-reshape','NumPy','vit navit naflex','numpy.reshape','Разверните два патча 2×2×3 в два вектора длины 12.',
      'x = np.arange(24).reshape(2,2,2,3)','out = np.reshape(x, (2, 12))','shape=(2,12), первая строка 0…11.','Reshape сохраняет порядок обхода элементов; смешивания пиксельных значений нет.','Число элементов должно сохраниться. Это ещё не обучаемый embedding.','assert out.shape==(2,12) and out[0].tolist()==list(range(12))')
    C('np-transpose','NumPy','raster finetune','numpy.transpose','Поменяйте HWC на CHW.',
      'x = np.zeros((2,3,4))','out = np.transpose(x, (2,0,1))','shape=(4,2,3).','Переставляются оси: C,H,W. Значения каналов не меняются местами RGB→BGR.','reshape(4,2,3) задаёт другой порядок соответствия элементов.','assert out.shape==(4,2,3)')
    C('np-stack','NumPy','blend','numpy.stack','Объедините два изображения в новый batch.',
      'images = [np.zeros((2,3)), np.ones((2,3))]','out = np.stack(images, axis=0)','shape=(2,2,3).','Stack добавляет новую ось. Все входы должны совпасть по форме.','concatenate по существующей оси не создаст batch такого вида.','assert out.shape==(2,2,3) and out[1].sum()==6')
    C('np-concat','NumPy','resnet unet','numpy.concatenate','Соедините признаки двух ветвей по каналам NHWC.',
      'a=np.zeros((1,2,2,3)); b=np.ones((1,2,2,5))','out = np.concatenate([a,b], axis=-1)','shape=(1,2,2,8).','Каналы сохраняются отдельными координатами, а не складываются.','Для residual sum нужны одинаковые формы, для concat — совпадение прочих осей.','assert out.shape==(1,2,2,8)')
    C('np-slice','NumPy','augmentation','basic slicing','Отразите HWC-картинку по горизонтали одной индексацией.',
      'x=np.arange(6).reshape(2,3,1)','out = x[:, ::-1, :]','Строки [[2,1,0],[5,4,3]], отрицательный stride.','Меняется ось ширины, а не строки или каналы. Обычно это view.','torch.from_numpy не принимает отрицательные strides: перед передачей сделайте copy.','assert out[:,:,0].tolist()==[[2,1,0],[5,4,3]]')
    C('np-copy','NumPy','augmentation','ndarray.copy','Создайте независимый crop, который можно менять.',
      'x=np.arange(9).reshape(3,3)','out = x[:2,:2].copy()','Изменение out не меняет x.','Обычный slice часто разделяет память с источником.','Присваивание crop=x[:2,:2] само по себе не копирует данные.','out[0,0]=99; assert x[0,0]==0')
    C('np-boolean','NumPy','seam-carving','boolean indexing','Удалите один заданный пиксель в каждой строке RGB.',
      'x=np.arange(18).reshape(2,3,3)\nkeep=np.array([[True,False,True],[False,True,True]])','out = x[keep].reshape(2,2,3)','shape=(2,2,3).','2D mask выбирает пространственные позиции, сохраняя по три канала каждой.','Нужен одинаковый остаток W−1 в каждой строке; иначе reshape скроет логическую ошибку.','assert out.tolist()==[[[0,1,2],[6,7,8]],[[12,13,14],[15,16,17]]]')
    C('np-where','NumPy','threshold','numpy.where','Примените порог >100 и выдайте значения 0/255.',
      'x=np.array([20,100,180])','out = np.where(x > 100, 255, 0)','[0,0,255].','Ровно 100 не проходит строгое неравенство.','where с тремя аргументами выбирает значения, а не координаты.','assert out.tolist()==[0,0,255]')
    C('np-nonzero','NumPy','sam-evaluation boxes','numpy.nonzero','Получите строки и столбцы foreground mask.',
      'mask=np.array([[False,True],[True,False]])','ys, xs = np.nonzero(mask)','ys=[0,1], xs=[1,0].','Порядок NumPy — строка(y), затем столбец(x). Для SAM точки собираются как x,y.','Пустая mask даст пустые массивы: min/max требуют проверки.','assert ys.tolist()==[0,1] and xs.tolist()==[1,0]')
    C('np-meshgrid','NumPy','geometry','numpy.meshgrid','Постройте координатные карты y,x формы 2×3.',
      'ys=np.arange(2); xs=np.arange(3)','yy, xx = np.meshgrid(ys, xs, indexing="ij")','Обе shape=(2,3), xx[1]=[0,1,2].','ij сохраняет порядок осей входов, поэтому первая ось — y.','xy и ij переставляют смысл первых двух осей.','assert yy.shape==(2,3) and xx[1].tolist()==[0,1,2]')
    C('np-bincount','NumPy','histogram meanshift','numpy.bincount','Посчитайте гистограмму четырёх целочисленных bins.',
      'bins=np.array([0,0,2,3,3,3])','out = np.bincount(bins, minlength=4)','[2,0,1,3].','minlength сохраняет отсутствующие bins нулями.','bincount требует неотрицательных integer labels, не float-интенсивностей.','assert out.tolist()==[2,0,1,3]')
    C('np-cumsum','NumPy','histogram','numpy.cumsum','Из частот получите накопленные counts.',
      'hist=np.array([2,0,1,3])','out = np.cumsum(hist)','[2,2,3,6].','Разделив на 6, получим CDF распределения; cumsum не нормирует сам.','Укажите ось для многомерных данных, если flatten нежелателен.','assert out.tolist()==[2,2,3,6]')
    C('np-mean','NumPy','whitebalance mobile','numpy.mean','Найдите среднее каждого канала HWC.',
      'x=np.array([[[1.,10.],[3.,30.]],[[5.,50.],[7.,70.]]])','out = np.mean(x, axis=(0,1), keepdims=True)','shape=(1,1,2), [4,40].','keepdims сохраняет оси для обратного broadcasting на H,W,C.','axis=-1 усреднил бы каналы, а не spatial-область.','np.testing.assert_allclose(out,[[[4,40]]])')
    C('np-einsum','NumPy','blend mobile','numpy.einsum','Для каждой RGB-тройки примените матрицу каналов.',
      'x=np.array([[[1.,2.,3.]]]); matrix=np.diag([2.,3.,4.])','out = np.einsum("hwc,dc->hwd", x, matrix)','[[[2,6,12]]].','Индекс c суммируется, d становится выходным каналом.','Это преобразование каналов, не пространственная свёртка.','assert out.tolist()==[[[2,6,12]]]')
    C('np-norm','NumPy','sift orb ratio clip','numpy.linalg.norm','Получите единичный descriptor [3,4].',
      'x=np.array([3.,4.])','out = x / np.linalg.norm(x)','[.6,.8].','Евклидова длина 5; нормировка сохраняет направление.','Для нулевого вектора нужно явно обработать знаменатель.','np.testing.assert_allclose(out,[.6,.8])')
    C('np-eigen','NumPy','harris','numpy.linalg.eigvalsh','Найдите собственные значения симметричной матрицы Harris.',
      'matrix=np.array([[100.,0.],[0.,1.]])','out = np.linalg.eigvalsh(matrix)','[1,100].','Один слабый eigenvalue соответствует направлению малого изменения — ребру.','Один большой eigenvalue не доказывает наличие угла.','assert out.tolist()==[1,100]')
    C('np-solve','NumPy','camera pnp poisson kalman image-pipeline','numpy.linalg.solve','Решите A x=b без явного inverse.',
      'A=np.array([[2.,0.],[0.,4.]]); b=np.array([6.,8.])','out = np.linalg.solve(A,b)','[3,2].','solve решает квадратную невырожденную систему; для overdetermined задачи есть lstsq.','Не заменяйте произвольную вырожденную систему обращением A.','np.testing.assert_allclose(out,[3,2])')
    C('np-argmin','NumPy','matching seam-carving','numpy.argmin','Выберите конец минимального пути по последней строке DP.',
      'cost=np.array([8.,3.,5.])','out = np.argmin(cost)','Столбец 1.','Индекс минимума — начало восстановления снизу вверх, не готовый весь шов.','Минимум каждой строки независимо может образовать несвязный путь.','assert out==1')
    C('np-default-rng','NumPy','experiment-control ransac','numpy.random.default_rng','Создайте локальный воспроизводимый генератор.',
      'seed=17','out = np.random.default_rng(seed)','Два генератора с seed=17 дают одинаковую последовательность при одинаковых вызовах.','Локальный Generator не обязан менять глобальный np.random state.','Seed не фиксирует порядок данных, если меняется последовательность вызовов.','assert (out.integers(100,size=8)==np.random.default_rng(17).integers(100,size=8)).all()')
    C('np-choice','NumPy','experiment-control sam-evaluation','Generator.choice','Выберите 4 разных индекса из 10.',
      'rng=np.random.default_rng(7)','out = rng.choice(10, size=4, replace=False)','Четыре различных индекса 0…9.','replace=False запрещает повтор одного индекса в данной выборке.','Это не гарантирует отсутствие пересечения двух независимо сделанных выборок.','assert len(set(out.tolist()))==4 and out.min()>=0 and out.max()<10')
    C('np-set-intersect','NumPy','pseudo-labels','numpy.intersect1d','Проверьте пересечение train/validation IDs.',
      'train=np.array([0,1,2]); valid=np.array([2,3])','out = np.intersect1d(train, valid)','[2]: обнаружена утечка.','Для корректного разделения результат должен быть пуст.','Разные имена переменных не доказывают разные изображения.','assert out.tolist()==[2]')
    C('np-any','NumPy','sam-auto','numpy.any','Объедините K масок перед подсчётом coverage.',
      'masks=np.array([[[1,0],[0,0]],[[1,1],[0,0]]],dtype=bool)','out = np.any(masks, axis=0)','[[True,True],[False,False]], coverage=0.5.','Перекрытый пиксель учитывается один раз.','sum площадей масок дал бы 3 вместо 2 уникальных пикселей.','assert out.sum()==2 and out.mean()==.5')
    C('np-logical-and','NumPy','mask-metrics iou','numpy.logical_and','Найдите пиксели пересечения двух bool-масок.',
      'a=np.array([1,1,0],bool); b=np.array([0,1,1],bool)','out = np.logical_and(a,b)','[False,True,False].','TP для бинарного foreground — сумма этого пересечения.','Intersection недостаточно для IoU: нужен union и правило для пустого union.','assert out.tolist()==[False,True,False]')
    C('np-fft','NumPy','fourier','numpy.fft.fft','Найдите DFT постоянного сигнала длины 4.',
      'x=np.ones(4)','out = np.fft.fft(x)','[4+0j,0,0,0].','Ненормированный forward FFT складывает постоянный уровень в DC-bin.','fftshift переставляет bins для показа, а не выполняет сам transform.','np.testing.assert_allclose(out,[4,0,0,0])')
    C('np-pad','NumPy','blur edges morphology dilated','numpy.pad','Дополните массив нулями на один элемент с каждой стороны.',
      'x=np.array([1,2,3])','out = np.pad(x, (1,1), mode="constant")','[0,1,2,3,0].','Padding задаёт граничное условие для фильтра.','reflect, edge и constant дают разные значения на границе.','assert out.tolist()==[0,1,2,3,0]')
    C('np-percentile','NumPy','matching','numpy.percentile','Найдите медиану score-значений.',
      'x=np.array([1.,2.,9.])','out = np.percentile(x, 50)','2.0.','Медиана меньше зависит от одного большого выброса, чем mean.','При чётном N результат может не совпадать ни с одним элементом.','assert out==2')

    C('pt-tensor','PyTorch','raster','torch.tensor','Создайте новый float tensor из values.',
      'values=[1,2,3]','out = torch.tensor(values, dtype=torch.float32)','float32 [1,2,3].','torch.tensor создаёт новое хранилище для этих данных.','Не путайте с from_numpy, который разделяет память.','assert out.dtype==torch.float32 and out.tolist()==values')
    C('pt-from-numpy','PyTorch','raster','torch.from_numpy','Создайте tensor, разделяющий CPU-память массива.',
      'array=np.array([1.,2.],dtype=np.float32)','out = torch.from_numpy(array)','Изменение out[0] видно в array[0].','Это zero-copy представление совместимого NumPy storage.','Изменение общего массива после forward может нарушить ожидаемые вычисления; при необходимости clone.','out[0]=9; assert array[0]==9')
    C('pt-full','PyTorch','caption-decoding bert','torch.full','Создайте стартовую BOS-последовательность [B,1].',
      'B=3; bos_id=1','out = torch.full((B,1), bos_id, dtype=torch.long)','[[1],[1],[1]].','Embedding ожидает целочисленные token IDs.','float BOS-тензор не годится для nn.Embedding.','assert out.dtype==torch.long and out.tolist()==[[1],[1],[1]]')
    C('pt-arange','PyTorch','position rope mrope','torch.arange','Создайте позиции нового блока после cache длины 5.',
      'past=5; length=3','out = torch.arange(past, past+length)','[5,6,7].','Позиции продолжаются, а не сбрасываются к нулю.','Одна и та же rotary-пара с разными позициями получает разные углы.','assert out.tolist()==[5,6,7]')
    C('pt-like','PyTorch','masks','torch.zeros_like','Создайте additive mask той же формы/device, но float32.',
      'template=torch.ones(2,3,dtype=torch.long)','out = torch.zeros_like(template, dtype=torch.float32)','[2,3] нулей float32.','Явно меняем dtype, сохраняем shape и device.','zeros_like без dtype унаследовал бы integer-тип template.','assert out.shape==(2,3) and out.dtype==torch.float32')
    C('pt-to','PyTorch','finetune','Tensor.to','Приведите tensor к float32 на CPU.',
      'x=torch.tensor([1,2],dtype=torch.long)','out = x.to(device="cpu", dtype=torch.float32)','CPU float32 [1,2].','to возвращает результат преобразования, исходная переменная x не меняет dtype.','to может вернуть тот же объект, если преобразование не нужно.','assert out.dtype==torch.float32 and x.dtype==torch.long')
    C('pt-permute','PyTorch','vit heads','Tensor.permute','Поменяйте [B,N,h,d] на [B,h,N,d].',
      'x=torch.arange(48).reshape(1,4,3,4)','out = x.permute(0,2,1,3)','shape=(1,3,4,4), обычно non-contiguous.','Головы становятся отдельной осью, значения не перемножаются.','После permute view может быть недопустим без contiguous.','assert out.shape==(1,3,4,4) and not out.is_contiguous()')
    C('pt-contiguous','PyTorch','heads','Tensor.contiguous','Сделайте переставленные оси contiguous.',
      'x=torch.arange(6).reshape(2,3).T','out = x.contiguous()','shape=(3,2), strides=(2,1).','Получаем привычное плотное расположение для последующего view.','contiguous не отменяет логическую транспозицию.','assert out.is_contiguous() and out.tolist()==[[0,3],[1,4],[2,5]]')
    C('pt-reshape','PyTorch','swin navit','torch.reshape','Сохраните значения non-contiguous tensor при flatten.',
      'x=torch.arange(6).reshape(2,3).T','out = torch.reshape(x, (6,))','[0,3,1,4,2,5].','reshape может сделать копию, если нужный вид невозможен через strides.','Нельзя считать reshape гарантированным zero-copy.','assert out.tolist()==[0,3,1,4,2,5]')
    C('pt-unsqueeze','PyTorch','sam-refinement','torch.unsqueeze','Добавьте batch-like ось к low-res logits H×W.',
      'logits=torch.zeros(256,256)','out = torch.unsqueeze(logits, 0)','shape=(1,256,256).','Добавляем одну ось без изменения значений.','Это меняет форму, но не превращает full-res final mask в правильные low-res logits.','assert out.shape==(1,256,256)')
    C('pt-squeeze','PyTorch','segmentation-task','Tensor.squeeze','Удалите только канальную singleton-ось, сохранив B=1.',
      'x=torch.zeros(1,1,2,3)','out = x.squeeze(1)','shape=(1,2,3).','Явная ось защищает batch от случайного удаления.','squeeze() без dim удалит и batch=1.','assert out.shape==(1,2,3)')
    C('pt-flatten','PyTorch','resnet','torch.flatten','Разверните все признаки после batch.',
      'x=torch.zeros(2,3,4,5)','out = torch.flatten(x, start_dim=1)','shape=(2,60).','Batch остаётся отдельным; 3·4·5 признаков одной картинки идут в строку.','flatten(x) без start_dim смешает и примеры батча.','assert out.shape==(2,60)')
    C('pt-cat','PyTorch','cache caption-cache unet','torch.cat','Добавьте новые keys к cached keys по оси последовательности.',
      'old=torch.zeros(1,2,5,4); new=torch.ones(1,2,1,4)','out = torch.cat([old,new], dim=2)','shape=(1,2,6,4).','Меняется длина K/V, а не число голов или head_dim.','stack добавил бы лишнюю ось и потребовал одинаковые формы.','assert out.shape==(1,2,6,4) and out[:,:,5,:].eq(1).all()')
    C('pt-stack','PyTorch','finetune','torch.stack','Соберите batch из двух CHW-картинок.',
      'items=[torch.zeros(3,4,4),torch.ones(3,4,4)]','out = torch.stack(items, dim=0)','shape=(2,3,4,4).','Добавляется batch-ось; каналы каждого изображения остаются вместе.','cat(items,dim=0) дал бы шесть каналов без batch.','assert out.shape==(2,3,4,4)')
    C('pt-expand','PyTorch','attention','Tensor.expand','Расширьте общий вектор на 3 строки без копирования.',
      'x=torch.tensor([[1.,2.]])','out = x.expand(3,-1)','shape=(3,2), stride первой оси 0.','Broadcast-view переиспользует память исходной строки.','Независимая in-place запись в строки expand небезопасна; для независимости нужна копия.','assert out.shape==(3,2) and out.stride(0)==0')
    C('pt-repeat','PyTorch','moe','Tensor.repeat','Создайте две копии вектора с самостоятельным storage результата.',
      'x=torch.tensor([1.,2.])','out = x.repeat(2,1)','[[1,2],[1,2]].','repeat материализует повторения, отличие от expand.','Это повторение данных, а не два эксперта с разными весами.','out[0,0]=7; assert x[0]==1 and out[1,0]==1')
    C('pt-matmul','PyTorch','attention image-retrieval clip siglip coca','torch.matmul','Сравните все Q текстов со всеми N image embeddings.',
      'text=torch.eye(2); images=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])','out = torch.matmul(text, images.T)','[[1,0,1],[0,1,1]], shape=(2,3).','Транспонирование делает внутренние размеры D,D совместимыми.','Без L2-нормировки это dot product, не обязательно cosine.','assert out.tolist()==[[1,0,1],[0,1,1]]')
    C('pt-einsum','PyTorch','heads','torch.einsum','Получите scores по головам из Q,K.',
      'q=torch.ones(1,2,3,4); k=torch.ones(1,2,5,4)','out = torch.einsum("bhnd,bhmd->bhnm",q,k)','shape=(1,2,3,5), все 4.','d суммируется, n и m сохраняются как query/key оси.','Здесь ещё нет деления на sqrt(d), маски и softmax.','assert out.shape==(1,2,3,5) and out.eq(4).all()')
    C('pt-tril','PyTorch','masks training','torch.tril','Создайте bool causal mask, True=разрешён key, включая текущую позицию.',
      'T=3','out = torch.tril(torch.ones(T,T,dtype=torch.bool))','[[1,0,0],[1,1,0],[1,1,1]].','Query i имеет право читать key j при j<=i. Диагональ разрешена.','У bool mask nn.MultiheadAttention True означает запрет, у SDPA True означает разрешение.','assert out.int().tolist()==[[1,0,0],[1,1,0],[1,1,1]]')
    C('pt-causal-offset','PyTorch','caption-cache','torch.le','Создайте causal mask для 2 новых queries после cache P=3.',
      'P=3; L=2\nqueries=P+torch.arange(L)[:,None]\nkeys=torch.arange(P+L)[None,:]','out = torch.le(keys, queries)','[[1,1,1,1,0],[1,1,1,1,1]].','Абсолютная позиция нового query i равна P+i.','Обычная tril(L,P+L) без offset закрыла бы почти весь прошлый cache.','assert out.int().tolist()==[[1,1,1,1,0],[1,1,1,1,1]]')
    C('pt-padding','PyTorch','masks training','torch.ne','Создайте mask валидных token IDs при PAD=0.',
      'ids=torch.tensor([[2,3,0],[4,0,0]])','out = ids.ne(0)','[[1,1,0],[1,0,0]].','Такая mask может обозначать допустимые keys; форму broadcast выбирают под API.','Loss ignore_index сам не закрывает attention keys.','assert out.int().tolist()==[[1,1,0],[1,0,0]]')
    C('pt-masked-fill','PyTorch','masks','Tensor.masked_fill','Закройте недопустимые scores перед softmax.',
      'scores=torch.tensor([[1.,2.,3.]])\nallowed=torch.tensor([[True,False,True]])','out = scores.masked_fill(~allowed, float("-inf"))','[1,−inf,3].','После softmax запрещённая позиция получит нулевой вес.','Полностью закрытая строка из −inf может дать NaN; нужен допустимый key или специальная обработка.','assert torch.isneginf(out[0,1]) and out[0,0]==1')
    C('pt-softmax','PyTorch','attention','torch.softmax','Нормируйте scores по keys, то есть последней оси.',
      'scores=torch.tensor([[0.,0.],[0.,0.]])','out = torch.softmax(scores, dim=-1)','Каждая строка [0.5,0.5].','Отдельное распределение на каждый query.','dim=0 смешивал бы queries вместо выбора keys для каждого query.','assert out.eq(.5).all() and out.sum(-1).eq(1).all()')
    C('pt-sdpa','PyTorch','attention masks','torch.nn.functional.scaled_dot_product_attention','Посчитайте causal attention с dropout=0.',
      'q=torch.ones(1,1,2,1); k=q.clone(); v=torch.tensor([[[[10.],[20.]]]])','out = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)','Первая query:10, вторая:15.','Одинаковые допустимые scores дают равные веса. Функция применяет dropout согласно dropout_p, независимо от внешнего model.eval.','Для inference явно задайте dropout_p=0; не переносите bool-смысл маски из MHA.','torch.testing.assert_close(out,torch.tensor([[[[10.],[15.]]]]))')
    C('pt-topk','PyTorch','image-retrieval moe map','torch.topk','Получите значения и индексы двух лучших scores.',
      'scores=torch.tensor([[.1,.9,.3,.7]])','out = torch.topk(scores,2,dim=-1)','indices=[[1,3]], values≈[[.9,.7]].','Индексы относятся к исходной оси кандидатов; значения — сами scores.','Равные значения не гарантируют устойчивый порядок tie-индексов.','assert out.indices.tolist()==[[1,3]]')
    C('pt-gather','PyTorch','classification','torch.gather','Для каждого примера выберите logit истинного класса.',
      'logits=torch.tensor([[2.,3.,4.],[5.,6.,7.]])\nlabels=torch.tensor([[2],[0]])','out = torch.gather(logits,1,labels)','[[4],[5]].','Index имеет shape [B,1], выбирает отдельный класс по каждой строке.','logits[:,labels] создаёт другое advanced-indexing поведение.','assert out.tolist()==[[4],[5]]')
    C('pt-argmax','PyTorch','caption-decoding','torch.argmax','Выберите следующий token ID из последних logits.',
      'last_logits=torch.tensor([[1.,3.,2.],[4.,0.,1.]])','out = torch.argmax(last_logits,dim=-1)','[1,0].','Softmax не нужен для argmax, поскольку он сохраняет порядок logits.','Argmax по всем осям уничтожит разделение примеров.','assert out.tolist()==[1,0]')
    C('pt-normalize','PyTorch','image-retrieval clip siglip ratio','torch.nn.functional.normalize','Нормируйте embedding [3,4] по последней оси.',
      'x=torch.tensor([[3.,4.]])','out = F.normalize(x,dim=-1)','[[.6,.8]].','Нормировка каждой строки делает dot product косинусным для ненулевых vectors.','Нулевой embedding не превращается в осмысленное единичное направление.','torch.testing.assert_close(out,torch.tensor([[.6,.8]]))')
    C('pt-roll','PyTorch','swin','torch.roll','Сдвиньте spatial-последовательность циклически на −1.',
      'x=torch.tensor([0,1,2,3])','out = torch.roll(x,shifts=-1,dims=0)','[1,2,3,0].','Roll переадресует элементы и заворачивает край.','Завёрнутый край не настоящий сосед в Swin: attention нуждается в mask.','assert out.tolist()==[1,2,3,0]')
    C('pt-unfold','PyTorch','swin vit conv','Tensor.unfold','Получите неперекрывающиеся окна длины 2.',
      'x=torch.arange(8)','out = x.unfold(0,2,2)','[[0,1],[2,3],[4,5],[6,7]].','Size=2, step=2 задают окна без overlap.','При step=1 окна перекрываются и могут разделять память.','assert out.tolist()==[[0,1],[2,3],[4,5],[6,7]]')
    C('pt-interpolate','PyTorch','unet segmentation-task retinanet','torch.nn.functional.interpolate','Увеличьте mask IDs nearest до 4×4.',
      'x=torch.tensor([[[[0.,1.],[2.,3.]]]])','out = F.interpolate(x,size=(4,4),mode="nearest")','shape=(1,1,4,4), ID из {0,1,2,3}.','Каждый ID копируется в ближайшую область, не усредняется с другим ID.','Bilinear по ID классов создал бы дробные метки.','assert out.shape==(1,1,4,4) and set(out.flatten().tolist())=={0,1,2,3}')
    C('pt-pool','PyTorch','resnet mobile moderncnn','torch.nn.functional.adaptive_avg_pool2d','Получите один средний признак на канал.',
      'x=torch.arange(8.).reshape(1,2,2,2)','out = F.adaptive_avg_pool2d(x,1)','shape=(1,2,1,1), значения 1.5 и 5.5.','Каждый канал усредняется spatial отдельно.','Не усредняйте каналы вместе, если нужен vector из C признаков.','torch.testing.assert_close(out.flatten(),torch.tensor([1.5,5.5]))')
    C('pt-grad-flag','PyTorch','init','Tensor.requires_grad_','Включите вычисление градиентов для leaf float tensor.',
      'x=torch.tensor([2.,3.])','out = x.requires_grad_(True)','x.requires_grad=True, leaf=True.','Autograd будет строить граф следующих операций с x.','Целочисленным token IDs градиент не назначают.','assert out.requires_grad and out.is_leaf')
    C('pt-backward','PyTorch','init finetune','Tensor.backward','Вычислите градиент суммы квадратов.',
      'x=torch.tensor([2.,3.],requires_grad=True)\nloss=(x*x).sum()','loss.backward()','x.grad=[4,6].','Для x² производная 2x. Scalar loss позволяет backward без внешнего вектора.','Градиенты накапливаются в .grad, а не заменяются автоматически.','assert x.grad.tolist()==[4,6]')
    C('pt-vjp','PyTorch','heads','torch.autograd.grad','Найдите vector-Jacobian product для y=x².',
      'x=torch.tensor([2.,3.],requires_grad=True)\ny=x.square(); weights=torch.tensor([1.,2.])','out = torch.autograd.grad(y,x,grad_outputs=weights)[0]','[4,12], x.grad остаётся None.','grad_outputs взвешивает выходы: (2x)·weights.','grad возвращает производную, но обычно не заполняет leaf .grad как backward.','assert out.tolist()==[4,12] and x.grad is None')
    C('pt-create-graph','PyTorch','init','torch.autograd.grad','Сохраните граф первой производной x³ для второй.',
      'x=torch.tensor(2.,requires_grad=True)\ny=x**3','out = torch.autograd.grad(y,x,create_graph=True)[0]','Первая производная 12, вторая производная доступна и тоже 12.','create_graph сохраняет differentiable операции производной.','retain_graph сохраняет исходный граф, но не эквивалентен create_graph.','assert out.item()==12 and torch.autograd.grad(out,x)[0].item()==12')
    C('pt-detach','PyTorch','embedding-projection','Tensor.detach','Отделите tensor от autograd без копии storage.',
      'x=torch.tensor([2.],requires_grad=True)\ny=x*3','out = y.detach()','out.requires_grad=False, память y и out общая.','Полезно для метрик и передачи признаков вне графа.','Изменения detach-view могут затронуть y; для независимой копии нужен clone.','assert not out.requires_grad and out.data_ptr()==y.data_ptr()')
    C('pt-clone','PyTorch','finetune','Tensor.clone','Скопируйте данные, сохранив связь autograd.',
      'x=torch.tensor([2.],requires_grad=True)','out = x.clone()','Storage другой, градиент через clone идёт к x.','clone differentiable, копирование данных не означает detach.','clone не делает автоматически новый независимый leaf для оптимизации.','assert out.data_ptr()!=x.data_ptr(); out.sum().backward(); assert x.grad.item()==1')
    C('pt-retain-grad','PyTorch','heads','Tensor.retain_grad','Сохраните .grad промежуточного non-leaf.',
      'x=torch.tensor(2.,requires_grad=True)\ny=x*3','y.retain_grad()','После (y*y).backward(): y.grad=12, x.grad=36.','Non-leaf .grad по умолчанию не сохраняется; retain_grad включает его хранение.','Это не тот же метод, что backward(retain_graph=True).','(y*y).backward(); assert y.grad.item()==12 and x.grad.item()==36')
    C('pt-no-grad','PyTorch','dropout finetune','torch.no_grad','Выполните вычисление без записи autograd.',
      'x=torch.tensor(2.,requires_grad=True)','with torch.no_grad():\n    out = x*3','out.requires_grad=False.','Контекст не строит граф обычных операций forward.','Не заменяет model.eval; dropout может оставаться случайным.','assert not out.requires_grad')
    C('pt-inference','PyTorch','image-retrieval','torch.inference_mode','Отключите autograd для inference-вычисления.',
      'x=torch.ones(2,requires_grad=True)','with torch.inference_mode():\n    out = x*2','out.requires_grad=False, inference tensor.','Режим дополнительно ограничивает служебный tracking по сравнению с no_grad.','Созданные inference tensors имеют ограничения при дальнейшей autograd-работе; это режим выдачи результата.','assert not out.requires_grad and torch.is_inference(out)')
    C('pt-eval','PyTorch','dropout finetune','torch.nn.Module.eval','Переключите dropout в inference-поведение.',
      'model=nn.Dropout(.9)\nx=torch.ones(5)','model.eval()','model(x) равен x.','eval рекурсивно устанавливает training=False у модулей.','Сам по себе eval не выключает autograd для Linear/Conv.','assert not model.training and model(x).eq(x).all()')
    C('pt-freeze','PyTorch','experiment-control','torch.nn.Module.requires_grad_','Заморозьте все параметры backbone.',
      'backbone=nn.Linear(3,4)','backbone.requires_grad_(False)','Все параметры requires_grad=False.','Общий вес остаётся в модели, но градиенты по нему не вычисляются.','Running statistics BatchNorm — буферы, поведение train/eval задаётся отдельно.','assert not any(p.requires_grad for p in backbone.parameters())')
    C('pt-numel','PyTorch','experiment-control','Tensor.numel','Подсчитайте обучаемые параметры Linear(3,4) с bias.',
      'model=nn.Linear(3,4)','out = sum(p.numel() for p in model.parameters() if p.requires_grad)','16 = 3·4+4.','Считаются элементы параметров, а не размер одного выхода батча.','state_dict содержит и buffers, их не надо безусловно считать обучаемыми.','assert out==16')
    C('pt-zero-grad','PyTorch','finetune','Optimizer.zero_grad','Сбросьте накопленные градиенты перед новым шагом.',
      'model=nn.Linear(2,1)\nopt=torch.optim.SGD(model.parameters(),lr=.1)\nmodel(torch.ones(1,2)).sum().backward()','opt.zero_grad(set_to_none=True)','p.grad становится None.','Обнуляются gradients параметров optimizer, а не значения самих weights.','Для gradient accumulation zero_grad вызывают на границе окна накопления.','assert all(p.grad is None for p in model.parameters())')
    C('pt-step','PyTorch','finetune','Optimizer.step','Сделайте шаг SGD с lr=0.1 по градиенту 4.',
      'p=nn.Parameter(torch.tensor(2.))\np.grad=torch.tensor(4.)\nopt=torch.optim.SGD([p],lr=.1)','opt.step()','p≈1.6.','SGD без momentum/decay вычитает lr·grad.','step не вычисляет градиент, это задача backward/grad.','torch.testing.assert_close(p,torch.tensor(1.6))')
    C('pt-clip-grad','PyTorch','amp-training focal','torch.nn.utils.clip_grad_norm_','Ограничьте общую L2-норму градиента до 1.',
      'p=nn.Parameter(torch.zeros(2)); p.grad=torch.tensor([3.,4.])','out = torch.nn.utils.clip_grad_norm_([p],1.0)','Возвращена прежняя норма 5; новый grad≈[.6,.8].','Метод меняет gradients in-place и возвращает норму до clipping.','При GradScaler сначала unscale_, затем clipping.','assert abs(out.item()-5)<1e-5; torch.testing.assert_close(p.grad,torch.tensor([.6,.8]))')
    C('pt-cross-entropy','PyTorch','classification captcha-heads','torch.nn.functional.cross_entropy','Посчитайте multiclass loss без предварительного softmax.',
      'logits=torch.tensor([[0.,0.]])\ntarget=torch.tensor([1])','out = F.cross_entropy(logits,target)','log(2)≈0.693147.','Target содержит integer class ID, class-ось у batch logits вторая.','Передача probabilities вместо logits меняет задачу.','assert abs(out.item()-np.log(2))<1e-6')
    C('pt-ignore','PyTorch','caption-metrics','torch.nn.functional.cross_entropy','Исключите PAD=0 из среднего token loss.',
      'logits=torch.zeros(2,3)\ntarget=torch.tensor([1,0])','out = F.cross_entropy(logits,target,ignore_index=0)','log(3), считается только одна валидная позиция.','Среднее берётся по неигнорируемым targets.','Батч целиком из ignore labels требует отдельной политики вместо NaN среднего.','assert abs(out.item()-np.log(3))<1e-6')
    C('pt-bce','PyTorch','segmentation-loss classification','torch.nn.functional.binary_cross_entropy_with_logits','Посчитайте binary loss для zero logits.',
      'logits=torch.zeros(2); target=torch.tensor([1.,0.])','out = F.binary_cross_entropy_with_logits(logits,target)','log(2)≈0.693147.','Встроенное устойчивое sigmoid+BCE ожидает float targets совпадающей формы.','Предварительный sigmoid не нужен.','assert abs(out.item()-np.log(2))<1e-6')
    C('pt-dice','PyTorch','segmentation-loss mask-metrics','Tensor.sum','Посчитайте soft Dice для p=[.8,.2], target=[1,0].',
      'p=torch.tensor([.8,.2]); target=torch.tensor([1.,0.])','out = 2*(p*target).sum()/(p.sum()+target.sum())','0.8, loss=0.2.','Это непрерывная мера overlap, до порога.','В общей функции добавьте epsilon и выберите оси/пустые случаи.','assert abs(out.item()-.8)<1e-6')
    C('pt-sigmoid','PyTorch','siglip','torch.sigmoid','Превратите независимые pair logits в scores.',
      'logits=torch.tensor([0.,0.])','out = torch.sigmoid(logits)','[.5,.5].','Каждая пара обрабатывается отдельно; для трёх нулевых logits сумма scores была бы 1.5.','Сумма случайно равна 1 здесь, но sigmoid не нормирует по кандидатам.','assert out.tolist()==[.5,.5]')
    C('pt-finished','PyTorch','caption-decoding','Tensor.logical_or','Обновите EOS-флаги, сохранив уже завершённых.',
      'finished=torch.tensor([True,False,False])\nnext_tokens=torch.tensor([7,2,8]); eos=2','out = finished.logical_or(next_tokens.eq(eos))','[True,True,False].','Новый не-EOS не должен обнулить ранее завершённый пример.','Остановить весь батч можно по out.all(), не out.any().','assert out.tolist()==[True,True,False]')
    C('pt-cos','PyTorch','rope mrope','torch.cos','Получите cos для rotary-углов 0 и pi/2.',
      'angles=torch.tensor([0.,torch.pi/2])','out = torch.cos(angles)','Приблизительно [1,0].','Углы задаются в радианах; на разных частотах позиция даёт разные углы.','Float-результат cos(pi/2) может быть малым ненулевым числом.','torch.testing.assert_close(out,torch.tensor([1.,0.]),atol=1e-6,rtol=0)')
    C('pt-layernorm','PyTorch','normalization adaptive','torch.nn.LayerNorm','Нормируйте каждый token по двум признакам без affine.',
      'x=torch.tensor([[1.,3.],[2.,6.]])\nlayer=nn.LayerNorm(2,elementwise_affine=False,eps=0.)','out = layer(x)','Обе строки [−1,1].','Статистика отдельно по последней оси, не по батчу.','eps=0 допустим здесь из-за ненулевой дисперсии; в обычной модели используйте стабилизирующий epsilon.','torch.testing.assert_close(out,torch.tensor([[-1.,1.],[-1.,1.]]))')
    C('pt-conv-groups','PyTorch','conv mobile moderncnn','torch.nn.Conv2d','Создайте depthwise 3×3 для 4 каналов без bias.',
      'channels=4','out = nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)','Вес shape=(4,1,3,3), 36 параметров.','Каждая группа читает один входной канал.','Обычная Conv4→4 имела бы 144 веса при groups=1.','assert out.weight.shape==(4,1,3,3) and out.weight.numel()==36')
    C('pt-embedding','PyTorch','bert position','torch.nn.Embedding','Создайте embedding table из 5 token IDs ширины 3.',
      'ids=torch.tensor([[1,2,0]])','out = nn.Embedding(5,3,padding_idx=0)(ids)','shape=(1,3,3), padding vector нулевой при стандартной инициализации.','Lookup выбирает строки таблицы по integer IDs.','padding_idx влияет на обновление этой строки, но не заменяет attention padding mask.','assert out.shape==(1,3,3) and out[0,2].eq(0).all()')
    C('pt-state-dict','PyTorch','finetune','torch.nn.Module.state_dict','Получите имена сохранённых weights и bias.',
      'model=nn.Linear(2,3)','out = model.state_dict()','Ключи weight и bias; weight.shape=(3,2).','state_dict хранит tensors параметров и persistent buffers.','Это не копия архитектуры и не обязательно независимая копия всех значений в памяти.','assert set(out)=={"weight","bias"} and out["weight"].shape==(3,2)')
    C('pt-load-state','PyTorch','finetune','torch.nn.Module.load_state_dict','Перенесите state_dict в такую же архитектуру.',
      'source=nn.Linear(2,3); target=nn.Linear(2,3)','out = target.load_state_dict(source.state_dict(), strict=True)','Совпали ключи и формы, веса target равны source.','Архитектуру создают до загрузки параметров.','strict=False не чинит произвольное несовпадение формы tensor.','torch.testing.assert_close(source.weight,target.weight)')
    C('pt-save-load','PyTorch','finetune','torch.load','Загрузите state_dict из memory-buffer на CPU.',
      'import io\nbuffer=io.BytesIO()\ntorch.save({"weight":torch.tensor([2.])},buffer)\nbuffer.seek(0)','out = torch.load(buffer,map_location="cpu",weights_only=True)','Словарь с CPU tensor weight=[2].','map_location выбирает устройство восстановления, weights_only ограничивает поддерживаемые объекты.','Загрузка dict сама не создаёт модель; её отдельно передают load_state_dict.','assert out["weight"].device.type=="cpu" and out["weight"].item()==2',doc='https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html')
    C('pt-dataset','PyTorch','experiment-control','torch.utils.data.TensorDataset','Свяжите изображения и targets по общему индексу.',
      'images=torch.zeros(4,3,2,2); labels=torch.arange(4)','out = torch.utils.data.TensorDataset(images,labels)','len=4; out[2][1]=2.','Оси числа примеров должны совпасть.','Dataset сам не создаёт train/validation split.','assert len(out)==4 and out[2][1].item()==2')
    C('pt-loader','PyTorch','captcha-heads','torch.utils.data.DataLoader','Соберите последовательные mini-batches по 2 примера.',
      'dataset=torch.utils.data.TensorDataset(torch.arange(5))','out = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)','Батчи [0,1], [2,3], [4].','Последний batch меньше, если drop_last=False.','Метрики нельзя усреднять по батчам без учёта их размеров.','assert [b[0].tolist() for b in out]==[[0,1],[2,3],[4]]')
    C('pt-topk-metric','PyTorch','metrics experiment-control','Tensor.eq','Проверьте попадание истинного класса в top-2.',
      'top_ids=torch.tensor([[1,3],[0,2]])\ntarget=torch.tensor([3,1])','out = top_ids.eq(target[:,None]).any(dim=1)','[True,False], accuracy=0.5.','Класс сравнивается с каждым кандидатом своей строки.','Top-k и exact строковой CAPTCHA измеряют разные объекты.','assert out.tolist()==[True,False]')
    C('pt-iou','PyTorch','mask-metrics iou sam-evaluation','torch.logical_or','Посчитайте binary-mask union.',
      'a=torch.tensor([1,1,0],dtype=torch.bool); b=torch.tensor([0,1,1],dtype=torch.bool)','out = torch.logical_or(a,b)','[True,True,True], union=3.','Для этих масок intersection=1, IoU=1/3.','Одинаковый foreground-числитель с другим знаменателем задаст другую метрику.','assert out.sum()==3')
    C('pt-distance','PyTorch','nms anchors boxes fcos','torch.clamp','Получите положительную ширину пересечения отрезков.',
      'left=torch.tensor([5.,2.]); right=torch.tensor([3.,7.])','out = torch.clamp(right-left,min=0)','[0,5].','Непересекающиеся интервалы должны дать нулевую, не отрицательную площадь.','Сначала согласуйте xyxy и непрерывные/включительные координаты.','assert out.tolist()==[0,5]')

    C('pt-roi-pool','PyTorch','rcnn','torch.nn.functional.adaptive_max_pool2d','Приведите уже вырезанную ROI feature map к 2×2 pooling.',
      'crop=torch.arange(16.).reshape(1,1,4,4)','out = F.adaptive_max_pool2d(crop,(2,2))','[[[[5,7],[13,15]]]].','Это учебная операция fixed-size pooling одного готового crop, не полная реализация ROI Pooling/ROIAlign с координатами.','Не считайте этот вызов доказательством эквивалентности ROIAlign.','assert out.tolist()==[[[[5,7],[13,15]]]]')
    C('pt-local-max','PyTorch','centernet','torch.nn.functional.max_pool2d','Получите максимум соседства 3×3 для сравнения с heatmap.',
      'heat=torch.zeros(1,1,3,3); heat[0,0,1,1]=.9','out = F.max_pool2d(heat,3,stride=1,padding=1)','Все девять позиций имеют максимум .9.','Сравнив heat==out, можно оставить локальные пики. В карточке вычисляется только максимум окрестности.','Это local-max операция heatmap, а не bbox-IoU NMS.','assert out.shape==heat.shape and out.eq(.9).all()')
    # Integration cards explicitly name external prerequisites and never download weights in checks.
    C('hf-model','Transformers','image-retrieval','AutoModel.from_pretrained','Загрузите модель из того же checkpoint, что и processor.',
      'from transformers import AutoModel\ncheckpoint="google/siglip2-so400m-patch16-naflex"',
      'model = AutoModel.from_pretrained(checkpoint).eval()', 'Загружены weights и config; модель в eval, autograd пока не отключён.',
      'Точная версия Transformers и доступная память должны поддерживать checkpoint. Первая загрузка скачивает веса, повторная может использовать cache.',
      'Не считайте device_map="auto" обязательным: он требует отдельной поддержки и может разнести weights по устройствам.',doc='https://huggingface.co/docs/transformers/model_doc/siglip2',runtime='external')
    C('hf-processor','Transformers','image-retrieval naflex','AutoProcessor.from_pretrained','Загрузите processor выбранного checkpoint.',
      'from transformers import AutoProcessor\ncheckpoint="google/siglip2-so400m-patch16-naflex"',
      'processor = AutoProcessor.from_pretrained(checkpoint)', 'Processor с согласованными настройками image/text preprocessing.',
      'Нормировка, resize, padding и NaFlex-аргументы следуют выбранной модели. Не подменяйте их случайным Resize(224).',
      'Процессор другого checkpoint может дать формально подходящий tensor, но неверную семантику входа.',doc='https://huggingface.co/docs/transformers/model_doc/siglip2',runtime='external')
    C('hf-preprocess','Transformers','image-retrieval','Processor.__call__','Подготовьте уже загруженные RGB images к image encoder.',
      '# Дано: processor нужного checkpoint, images — список PIL RGB images.',
      'inputs = processor(images=images, return_tensors="pt")','Словарь tensor-входов; NaFlex может включать masks и spatial_shapes.',
      'Не предполагайте, что для каждой модели достаточно одного pixel_values. Сохраняйте все требуемые ключи.',
      'Это preprocessing, а не forward и не embedding.',doc='https://huggingface.co/docs/transformers/model_doc/siglip2',runtime='external')
    C('hf-features','Transformers','image-retrieval','Model.get_image_features','Получите image features и учтите различие версий API.',
      '# Дано: model в eval, inputs на согласованном устройстве.',
      'with torch.inference_mode():\n    features = model.get_image_features(**inputs)\n    embeddings = features if isinstance(features, torch.Tensor) else features.pooler_output',
      'Ожидается один embedding на изображение, [B,D]; проверьте контракт установленной версии.',
      'В семинаре используется ModelOutput.pooler_output, в других версиях get_image_features возвращает Tensor. Проверка типа делает различие явным.',
      'Не делайте повторный pooling неизвестного выхода и не считайте полную форму patch states равной [B,D].',doc='https://huggingface.co/docs/transformers/model_doc/siglip2',runtime='external')
    C('sam-set-image','SAM','sam-prompts','SamPredictor.set_image','Закешируйте embedding текущего RGB-изображения.',
      '# Дано: predictor=SamPredictor(уже загруженный SAM), image_rgb uint8 HWC.',
      'predictor.set_image(image_rgb)','Готов image embedding для повторных prompts этого изображения.',
      'Подсказки меняются без повторного set_image, пока картинка та же. Новый кадр требует нового embedding.',
      'Не подавайте BGR как RGB и не оставляйте cache от другой картинки.',doc='https://github.com/facebookresearch/segment-anything',runtime='external')
    C('sam-predict','SAM','sam-prompts','SamPredictor.predict','Запросите неоднозначные маски по одной foreground-точке.',
      '# Дано: predictor.set_image уже выполнен для текущего изображения.\npoint=np.array([[30,40]]); labels=np.array([1])',
      'masks, scores, logits = predictor.predict(point_coords=point,point_labels=labels,multimask_output=True)',
      'Кандидатные masks, predicted quality scores и low-res logits.',
      'Точка задана x=30,y=40. Три правдоподобных маски могут соответствовать части и целому.',
      'Наибольший predicted score не гарантирует нужную гранулярность объекта.',doc='https://github.com/facebookresearch/segment-anything',runtime='external')
    C('sam-mask-input','SAM','sam-refinement','SamPredictor.predict(mask_input)','Уточните выбранную маску её low-res logits.',
      '# Дано: logits от предыдущего predict, best — индекс выбранной маски.\n# point/labels заданы для текущего изображения.',
      'masks, scores, _ = predictor.predict(point_coords=point,point_labels=labels,mask_input=logits[best][None],multimask_output=False)',
      'Одна уточнённая маска при согласованных prompts.',
      'Дополнительная ось даёт ожидаемую форму mask_input [1,256,256] у стандартного SAM.',
      'Final bool mask исходного размера не заменяет logits без проверки контракта.',doc='https://github.com/facebookresearch/segment-anything',runtime='external')
    C('sam-generator','SAM','sam-auto','SamAutomaticMaskGenerator','Создайте генератор с сеткой 16×16 prompts на базовом crop.',
      'from segment_anything import SamAutomaticMaskGenerator\n# sam — уже загруженная модель.',
      'generator = SamAutomaticMaskGenerator(sam,points_per_side=16,pred_iou_thresh=0.88)',
      'Настроен генератор; маски появятся после generator.generate(image_rgb).',
      '16 — сторона сетки, примерно 256 точек до учёта дополнительных crops и фильтрации.',
      'Порог predicted IoU — фильтр оценённого качества, не измеренный IoU с GT.',doc='https://github.com/facebookresearch/segment-anything',runtime='external')
    C('cuda-sync','PyTorch','sam2-comparison detector-workflow','torch.cuda.synchronize','Дождитесь завершения GPU-работы перед/после замера.',
      '# Требуется CUDA. Тот же device должен использоваться внутри измеряемого forward.',
      'torch.cuda.synchronize(device)', 'Все ранее поставленные задачи данного CUDA device завершены.',
      'GPU асинхронен относительно CPU; синхронизация задаёт границу временного интервала. Перед измерением нужен прогрев.',
      'Не вызывайте CUDA API на CPU-only машине и не включайте синхронизацию только у одной сравниваемой модели.',runtime='external')
    C('torch-compile','PyTorch','caption-decoding amp-training','torch.compile','Создайте compiled вариант функции decode_step.',
      '# decode_step — существующая функция; требуется поддерживаемая среда compiler.',
      'compiled_step = torch.compile(decode_step, dynamic=True)', 'Callable, который компилирует поддерживаемый путь при вызовах.',
      'Время первого вызова содержит подготовку. Изменяющиеся shapes/ветвления могут вызывать graph breaks или recompilation.',
      'Compile не исправляет causal mask и не гарантирует ускорение любого кода.',runtime='external')
    C('amp-autocast','PyTorch','amp-training','torch.autocast','Оберните CUDA forward и loss в autocast.',
      '# Требуется CUDA; model,x,target,criterion уже определены.',
      'with torch.autocast(device_type="cuda", dtype=torch.float16):\n    loss = criterion(model(x), target)',
      'Подходящие операции выбирают reduced precision, параметры не обязаны все стать float16.',
      'Backward обычно выполняют после выхода из autocast-контекста. FP16 scaling настраивается отдельно.',
      'Не переводите blindly все индексы и labels в half.',runtime='external')
    C('amp-unscale','PyTorch','amp-training','GradScaler.unscale_','Верните настоящий масштаб gradients до clipping.',
      '# scaler.scale(loss).backward() уже выполнен; optimizer содержит те же параметры.',
      'scaler.unscale_(optimizer)', 'Градиенты optimizer возвращены к исходному масштабу для clipping/анализа.',
      'После этого можно clip_grad_norm_, затем scaler.step и scaler.update. В карточке фокус только на unscale.',
      'Не вызывайте повторный unscale_ для того же optimizer между одним backward-окном и step.',doc='https://docs.pytorch.org/docs/2.14/notes/amp_examples.html',runtime='external')
    C('yolo-load','Ultralytics','detector-workflow','YOLO','Откройте доверенный локальный checkpoint детектора.',
      'from ultralytics import YOLO\ncheckpoint="my_detector.pt"  # существующий локальный файл',
      'model = YOLO(checkpoint)', 'Модель и метаданные выбранной версии; это ещё не обучение и не export.',
      'Файл должен соответствовать поддерживаемой версии Ultralytics. Выбор weights задаёт конкретную архитектуру.',
      'Не переносите описание head от другой версии YOLO по одному общему названию.',doc='https://docs.ultralytics.com/usage/python/',runtime='external')
    C('yolo-export','Ultralytics','detector-workflow','YOLO.export','Экспортируйте уже загруженную модель в ONNX.',
      '# model — уже загруженная YOLO, нужны зависимости ONNX export.',
      'path = model.export(format="onnx",imgsz=640)', 'Путь экспортированного артефакта при успешном export.',
      'После экспорта отдельно сравнивают outputs и latency на одинаковых inputs/device.',
      'Сам факт появления ONNX не доказывает ускорение или точное совпадение постобработки.',doc='https://docs.ultralytics.com/modes/export/',runtime='external')

    # Cover related lecture topics without claiming the API call itself was in every notebook.
    extra={'pt-distance':'nms anchors boxes fcos ssd yolo','pt-matmul':'coca navit naflex detr',
      'np-solve':'pnp camera kalman poisson','np-pad':'classicaldetectors blur edges morphology dilated',
      'pt-conv-groups':'mobile moderncnn conv','pt-pool':'retinanet','np-norm':'sift orb ratio',
      'np-any':'sam-auto pseudo-labels','pt-inference':'sam2-comparison','np-astype':'image-pipeline'}
    for card in cards:card['topics']=list(dict.fromkeys(card['topics']+extra.get(card['id'],'').split()))
    for t in topics:
        matches=[c for c in cards if t['id'] in c['topics']]
        if not matches:raise ValueError('No code card for '+t['id'])
        t['codeCards']=[c['id'] for c in matches]
        if t.get('supplemental'):
            c=next((c for c in matches if c['runtime']=='cpu'),matches[0])
            t['implementation']=dict(language='python',code='import numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n'+c['setup']+'\n'+c['code'],note=c['explanation']+' '+c['result'],source=deepcopy(t['source']))
    return cards
