"""Lecture-grounded mathematical notes and short, independently authored Python examples."""
from copy import deepcopy
from textwrap import dedent


def extend(topics, questions):
    by_id={t['id']:t for t in topics}
    def N(key, math, code, note, cells=None):
        topic=by_id[key]
        topic['mathNotes']=dedent(math).strip()
        source=deepcopy(topic['source'])
        if cells is not None:source['cells']=cells
        topic['implementation']={'language':'python','code':dedent(code).strip(),
                                 'note':note,'source':source}

    N('attention',r'''
    Для одной головы $Q\in\mathbb R^{n\times d_k}$, $K\in\mathbb R^{m\times d_k}$,
    $V\in\mathbb R^{m\times d_v}$. Softmax применяется **по ключам**, отдельно для каждой строки:
    $$S=\frac{QK^\top}{\sqrt{d_k}},\qquad A_{ij}=\frac{e^{S_{ij}}}{\sum_{r=1}^{m}e^{S_{ir}}},\qquad Y=AV\in\mathbb R^{n\times d_v}.$$
    Поэтому $\sum_j A_{ij}=1$, а каждый выход — взвешенная сумма values.

    **Откуда масштаб.** При независимых компонентах с нулевым средним, единичной дисперсией
    и независимости произведений по координатам:
    $$\operatorname{Var}\!\left(\sum_{r=1}^{d_k}q_rk_r\right)=\sum_{r=1}^{d_k}\operatorname{Var}(q_rk_r)=d_k,
    \qquad \operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)=1.$$
    Это обоснование масштаба при указанных допущениях, а не утверждение, что обученные Q и K всегда независимы.
    При $d_k=64$ делитель равен $8$. Вычитание максимума из строки logits не меняет softmax и повышает устойчивость.
    ''','''
    import math
    import torch

    def attention(q, k, v, allowed=None):
        # q: [..., n, dk], k: [..., m, dk], v: [..., m, dv]
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if allowed is not None:  # True = разрешено
            if not allowed.any(dim=-1).all():
                raise ValueError("Каждой query нужен хотя бы один key")
            scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v, weights
    # y, weights = attention(q, k, v)
    ''','Схема из ячеек 4–7: без dropout и проекций Q/K/V. Полностью замаскированные строки отклоняются явно.',[4,5,6,7])

    N('heads',r'''
    Для $h$ голов обычно $d_k=d_v=d/h$:
    $$\mathrm{head}_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V),\qquad
    \operatorname{MHA}(X)=\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O.$$
    $$\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2,\qquad
    P_{\rm FFN}=2dd_{\rm ff}\quad\text{без bias}.$$
    Attention смешивает позиции, FFN — признаки внутри каждой позиции. При $d=64,d_{\rm ff}=256$ получаем $32768$ весов.

    **Уточнение к обоснованию в лекции.** $\operatorname{rank}(QK^\top)\leq d_k$ относится к logits;
    нелинейный построчный softmax не обязан сохранять этот ранг. Контрпример при $d_k=1$:
    $$S=\begin{pmatrix}0&0\\0&1\end{pmatrix},\quad
    \operatorname{softmax}(S)=\begin{pmatrix}1/2&1/2\\1/(1+e)&e/(1+e)\end{pmatrix},\quad
    \det=\frac{e-1}{2(1+e)}\ne0.$$
    Поэтому нельзя переносить оценку ранга logits на матрицу весов внимания.
    ''','''
    import torch
    from torch import nn

    class EncoderBlock(nn.Module):
        def __init__(self, d=64, heads=4, dff=256):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))
            self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

        def forward(self, x):  # [B, N, d], вариант pre-norm
            z = self.norm1(x)
            x = x + self.attn(z, z, z, need_weights=False)[0]
            return x + self.ffn(self.norm2(x))
    # block = EncoderBlock(); y = block(torch.randn(2, 10, 64))
    ''','Учебный encoder block по компонентам лекции. Здесь явно выбран pre-norm; dropout опущен. Это не полная модель.',[7,8,10,11,12,13])

    N('masks',r'''
    Удобно различать булеву допустимость $B_{ij}$ и аддитивную маску $M_{ij}$:
    $$M_{ij}=\begin{cases}0,&B_{ij}=1,\\-\infty,&B_{ij}=0,\end{cases}\qquad
    A=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M).$$
    Для causal attention $B_{ij}=\mathbf1[j\leq i]$. Диагональ разрешена: скрытое состояние позиции $i$
    читает текущий входной токен, а target для обычного CLM — **следующий** токен.
    При длине $T$ разрешено $T(T+1)/2$ пар.

    Label mask относится к loss, а не к доступному контексту:
    $$\mathcal L=-\frac{1}{|\mathcal S|}\sum_{i\in\mathcal S}\log p_\theta(y_i\mid\text{доступный контекст}).$$
    Позиции вне $\mathcal S$ могут оставаться видимыми в attention. При $|\mathcal S|=0$ среднее не определено.
    ''','''
    import torch
    import torch.nn.functional as F

    def causal_allowed(lengths, T):
        pos = torch.arange(T, device=lengths.device)
        causal = pos[:, None] >= pos[None, :]
        valid_keys = pos[None, :] < lengths[:, None]
        return causal[None, None] & valid_keys[:, None, None, :]

    def masked_token_loss(logits, labels):
        # logits: [B,T,V]; labels: [B,T], исключённые targets = -100
        if not (labels != -100).any():
            raise ValueError("Нет targets для усреднения")
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    ''','В causal_allowed True означает разрешение. У nn.MultiheadAttention булевы маски имеют другое соглашение: проверяйте API. Padded queries исключаются из loss отдельно; длины здесь должны быть положительными.',[8,9,25])

    N('moe',r'''
    Router выдаёт $r(x)=W_gx\in\mathbb R^E$. Для выбранного множества $\mathcal T(x)=\operatorname{TopK}(r(x))$:
    $$g_i(x)=\frac{e^{r_i(x)}}{\sum_{j\in\mathcal T(x)}e^{r_j(x)}}\mathbf1[i\in\mathcal T(x)],\qquad
    y(x)=\sum_{i\in\mathcal T(x)}g_i(x)E_i(x).$$
    Здесь нормировка выполняется по выбранным экспертам. Зануление остальных **logits** перед softmax не исключает их: $e^0=1$.
    При одинаковых FFN без bias $P_E=2dd_{\rm ff}$, всего экспертных весов $EP_E$, активно на токен $kP_E$;
    общие слои, router и коммуникации считаются отдельно.

    В приведённом в лекции вспомогательном loss:
    $$\mathcal L_{\rm aux}=\alpha E\sum_{i=1}^E f_i\bar p_i.$$
    Нужно согласовать определение доли назначений $f_i$ с top-k; формула сама по себе не гарантирует равенства нагрузок каждого батча.
    ''','''
    import torch
    from torch import nn

    class TinyMoE(nn.Module):
        def __init__(self, d=32, dff=64, experts=4, k=2):
            super().__init__()
            assert 1 <= k <= experts
            self.k = k
            self.router = nn.Linear(d, experts, bias=False)
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))
                for _ in range(experts)])

        def forward(self, x):  # [T,d]
            scores, ids = self.router(x).topk(self.k, dim=-1)
            gates = scores.softmax(dim=-1)
            out = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                token, slot = torch.where(ids == i)
                if token.numel():
                    out = out.index_add(0, token, expert(x[token]) * gates[token, slot, None])
            return out
    ''','Прозрачная top-k схема по ячейкам 14–15. Без capacity, distributed routing и auxiliary loss; цикл выбран для читаемости.',[14,15])

    N('position',r'''
    Learned PE хранит $P\in\mathbb R^{L_{\max}\times d}$, и $z_i=x_i+P_i$;
    число параметров равно $L_{\max}d$.
    Для чётного $d$ синусоидальное кодирование из лекции:
    $$\operatorname{PE}(p,2i)=\sin(p\omega_i),\qquad
    \operatorname{PE}(p,2i+1)=\cos(p\omega_i),\qquad \omega_i=10000^{-2i/d}.$$
    Из формулы сложения $\sin((p+\Delta)\omega)$ следует линейная связь двух координат пары при фиксированном $\Delta$.
    Это объясняет структуру относительного сдвига, но не доказывает качество на любой длине.
    Без позиций и структурных масок полное self-attention **эквивариантно**:
    $$\operatorname{SA}(\Pi X)=\Pi\operatorname{SA}(X),$$
    а не инвариантно: выходные строки переставляются вместе со входом.
    ''','''
    import math
    import torch

    def sinusoidal_pe(length, d):
        assert d % 2 == 0
        pos = torch.arange(length, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(length, d)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return pe
    # z = x + sinusoidal_pe(x.shape[1], x.shape[2]).to(x)[None]
    ''','Чётная ширина указана явно. Код строит позиции на CPU и переносит их к x в примере использования.',[16,18,19,20,21])

    N('rope',r'''
    Пусть $R_p$ — блочно-диагональный поворот с блоками
    $$R(p\omega_i)=\begin{pmatrix}\cos(p\omega_i)&-\sin(p\omega_i)\\
    \sin(p\omega_i)&\cos(p\omega_i)\end{pmatrix},\qquad \omega_i=10000^{-2i/d}.$$
    Тогда $\tilde q_m=R_mq_m$, $\tilde k_n=R_nk_n$ и
    $$\tilde q_m^\top\tilde k_n=q_m^\top R_m^\top R_nk_n=q_m^\top R_{n-m}k_n.$$
    Позиционный множитель зависит от разности индексов; содержательные $q_m,k_n$ всё ещё влияют на score.
    Из $R_p^\top R_p=I$ следует сохранение нормы. При общем сдвиге $m,n$ разность не меняется,
    если сравниваемые содержательные векторы фиксированы. Вычислимость вращения не обещает бесконечного обобщения.
    ''','''
    import torch

    def rope(x, positions, base=10000.0):
        # x: [...,T,d], positions: [T]; вращаем соседние пары
        d = x.shape[-1]
        assert d % 2 == 0
        freq = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
        angle = positions.to(x.device).float()[:, None] * freq[None, :]
        c, s = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a*c - b*s, a*s + b*c), dim=-1).flatten(-2)
    # q_rot, k_rot = rope(q, pos), rope(k, pos)
    ''','Иллюстрация соглашения с соседними парами. Некоторые модели используют split-half layout: их pretrained веса нельзя смешивать с другим layout без согласования.',[22,23])

    N('mrope',r'''
    В лекции rotary-каналы разделены между временем и двумя spatial-осями:
    $$d=d_t+d_h+d_w,\qquad
    R(t,h,w)=\operatorname{diag}(R_t(t),R_h(h),R_w(w)).$$
    В парной реализации ширина каждого раздела чётная. Для фиксированных $q,k$:
    $$\tilde q_a^\top\tilde k_b=\sum_{u\in\{t,h,w\}}q_{a,u}^\top R_u(p_{b,u}-p_{a,u})k_{b,u}.$$
    Горизонтальное и вертикальное смещения различаются даже при близких индексах flatten-потока.
    У неподвижного изображения время может быть общим для всех патчей. Распределение каналов и позиций текста
    является соглашением конкретной модели, а не универсальным равным делением на три.
    ''','''
    import torch

    def mrope(x, coords, widths, rotate):
        # coords: [T,3] с колонками t,h,w; rotate — функция rope выше
        assert sum(widths) == x.shape[-1]
        assert all(w > 0 and w % 2 == 0 for w in widths)
        chunks = torch.split(x, widths, dim=-1)
        return torch.cat([rotate(part, coords[:, axis])
                          for axis, part in enumerate(chunks)], dim=-1)
    # y = mrope(q, coords, (4, 6, 6), rope)  # q[...,16]
    ''','Схема осевого разбиения, а не реализация конкретного Qwen checkpoint. rotate передаётся явно; её определение приведено в теме RoPE.',[24])

    N('bert',r'''
    Для выбранных MLM-позиций $\mathcal M$ target — исходный токен $x_i$, вход — испорченная последовательность $\tilde x$:
    $$\mathcal L_{\rm MLM}=-\frac1{|\mathcal M|}\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).$$
    Из выбора $15\%$ позиций и замены $80\%$ выбранных на MASK следует доля $0.15\cdot0.8=0.12$ **всех** токенов.
    Случайная замена $10\%$ выбранных даёт $1.5\%$ всех токенов. Неизменённые выбранные позиции тоже имеют loss.

    Для приближённого продолжения можно запросить
    $$p_\theta(x_{T+1}\mid x_1,\ldots,x_T,[\mathrm{MASK}]).$$
    Добавляем **новую** MASK-позицию, сохраняя последний известный токен. Это применение MLM;
    оно не превращает BERT в causal LM и не даёт автоматически корректного авторегрессионного joint distribution.
    ''','''
    import torch
    import torch.nn.functional as F

    def mlm_loss(logits, original_ids, selected):
        # logits от BERT на испорченном входе: [B,T,V]
        if not selected.any():
            raise ValueError("Нужна хотя бы одна MLM-позиция")
        labels = original_ids.masked_fill(~selected, -100)
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    # loss = mlm_loss(model(corrupted_ids).logits, original_ids, selected)
    ''','Схема loss по MLM. Спецтокены и padding исключаются при построении selected; генерация corrupted_ids здесь не реализована.',[25,29])

    N('training',r'''
    Авторегрессионная факторизация и teacher forcing:
    $$p_\theta(x_{1:T})=\prod_{t=1}^T p_\theta(x_t\mid x_{<t}),\qquad
    \mathcal L_{\rm CLM}=-\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{\leq t}).$$
    Все правильные входные токены известны на train, поэтому causal-маска позволяет один параллельный forward.
    На generation следующий input ещё нужно выбрать, поэтому шаги зависимы.
    В packing независимых документов маска может быть
    $$B_{ij}=\mathbf1[g_i=g_j]\mathbf1[j\leq i],$$
    где $g_i$ — идентификатор документа. На границе также исключают ошибочный target «конец A → начало B»;
    attention mask сама эту целевую пару не исправляет.
    ''','''
    import torch.nn.functional as F

    def clm_loss(logits, tokens, doc_ids):
        # logits [B,T,V], tokens/doc_ids [B,T]; -1 означает padding
        labels = tokens[:, 1:].clone()
        same_doc = (doc_ids[:, :-1] == doc_ids[:, 1:]) & (doc_ids[:, 1:] >= 0)
        labels[~same_doc] = -100
        if not same_doc.any():
            raise ValueError("Нет targets внутри документов")
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), ignore_index=-100)
    ''','Ручной сдвиг logits/targets ровно один раз. Не передавайте уже сдвинутые labels в модель, которая сама выполняет такой же сдвиг. Packing attention mask задаётся отдельно.',[25])

    N('cache',r'''
    Для текущего causal-префикса в каждом слое храним прошлые проекции:
    $$K_{1:t}=[K_{1:t-1};k_t],\quad V_{1:t}=[V_{1:t-1};v_t],\quad
    y_t=\operatorname{softmax}(q_tK_{1:t}^{\top}/\sqrt{d_k})V_{1:t}.$$
    Новые токены не меняют контекст старых позиций, поэтому их K/V можно переиспользовать.
    Новый query всё ещё читает $t$ ключей: стоимость attention шага $\mathcal O(td)$, а не константа.
    Для обычного MHA cache порядка $2BLTd$ элементов по $L$ слоям; GQA/MQA имеют другое число KV-голов.
    Сумма пар по генерации $\sum_{t=1}^Tt=T(T+1)/2$, тогда как полный пересчёт всех префиксов даёт $\sum_t t^2=\mathcal O(T^3)$.
    ''','''
    import math
    import torch

    def cached_step(q_new, k_new, v_new, cache=None):
        # ровно одна новая позиция; тензоры [B,h,1,dk]
        assert q_new.shape[-2] == k_new.shape[-2] == v_new.shape[-2] == 1
        k, v = k_new, v_new
        if cache is not None:
            k = torch.cat((cache[0], k_new), dim=-2)
            v = torch.cat((cache[1], v_new), dim=-2)
        weights = (q_new @ k.transpose(-2, -1) / math.sqrt(q_new.shape[-1])).softmax(-1)
        return weights @ v, (k, v)
    ''','Один шаг одной головы/слоя после QKV-проекций. Предполагаются согласованные позиции и тот же префикс. Для нескольких новых токенов нужна отдельная causal-маска; для скорости cat обычно заменяют буфером.',[26,27])

    N('vit',r'''
    Для изображения $H\times W\times C$ и патча $P\times P$ при делимости сторон:
    $$N=\frac{HW}{P^2},\quad x_p\in\mathbb R^{N\times(P^2C)},\quad
    z_0=[x_{\rm cls};x_pE]+E_{\rm pos},\quad E\in\mathbb R^{P^2C\times d}.$$
    Projection общая для всех патчей: $P^2Cd$ весов без bias, а learned PE с CLS имеет $(N+1)d$ параметров.
    Conv2d с kernel=stride=$P$ выполняет то же линейное преобразование каждого неперекрывающегося патча
    при согласованном порядке координат.
    $$\#\mathrm{scores}=(N+1)^2\quad\text{на голову},\qquad \text{стоимость пар}\sim\mathcal O(N^2d).$$
    Удвоение обеих сторон даёт примерно $4N$ патчей и $16$-кратный рост парных операций, но не всех FLOPs модели.
    ''','''
    from torch import nn

    class PatchEmbedding(nn.Module):
        def __init__(self, patch=16, channels=3, d=64):
            super().__init__()
            self.patch = patch
            self.proj = nn.Conv2d(channels, d, patch, stride=patch)

        def forward(self, image):  # [B,C,H,W]
            H, W = image.shape[-2:]
            assert H % self.patch == W % self.patch == 0
            return self.proj(image).flatten(2).transpose(1, 2)  # [B,N,d]
    # tokens = PatchEmbedding()(image)  # CLS и PE добавляются далее
    ''','Сокращённый patch embedding по ячейке 31. CLS, PE, encoder и классификационная голова в этот класс не входят.',[30,31])

    N('swin',r'''
    Пусть $N=HW$ — число **токенов текущей карты**, сторона окна $M$, ширина $C$.
    В окне $M^2$ токенов, матрица одной головы содержит $M^4$ пар, окон $N/M^2$:
    $$\#\mathrm{pairs}=\frac{N}{M^2}M^4=NM^2,\qquad
    \Omega(\mathrm{W\!\!\!-MSA})=4NC^2+2NM^2C.$$
    Для global MSA парный член $2N^2C$. Это оценка attention-блока из лекции; FFN и аппаратные накладные расходы считаются отдельно.

    При SW-MSA сдвиг $s=\lfloor M/2\rfloor$ меняет группы, после attention выполняется обратный сдвиг.
    У cyclic shift маска удаляет ложные связи противоположных краёв:
    $$A=\operatorname{softmax}(QK^\top/\sqrt{d_k}+B_{\rm rel}+M_{\rm shift}),\quad
    \#B_{\rm table}=(2M-1)^2h.$$
    Для $M=7,h=6$ это $1014$ параметров таблиц. Patch merging:
    $$H\times W\times C\;\longrightarrow\;(H/2)\times(W/2)\times4C
    \;\xrightarrow{\rm linear}\;(H/2)\times(W/2)\times2C.$$
    Одна смена окон не даёт мгновенного полного global attention. Если добавлен padding и требуется исключить его вклад,
    нужна соответствующая маска; нельзя считать это автоматически выполненным любой реализацией.
    ''','''
    import torch
    from torch import nn

    def window_partition(x, M):  # x: [B,H,W,C], без padding
        B, H, W, C = x.shape
        assert H % M == W % M == 0
        return x.reshape(B, H//M, M, W//M, M, C).permute(0,1,3,2,4,5).reshape(-1,M*M,C)

    class PatchMerging(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm = nn.LayerNorm(4 * channels)
            self.proj = nn.Linear(4 * channels, 2 * channels, bias=False)

        def forward(self, x):
            assert x.shape[1] % 2 == x.shape[2] % 2 == 0
            parts = [x[:, i::2, j::2] for i, j in [(0,0),(1,0),(0,1),(1,1)]]
            return self.proj(self.norm(torch.cat(parts, dim=-1)))
    # Перед SW-MSA: torch.roll(x, shifts=(-s,-s), dims=(1,2))
    # После partition attention ОБЯЗАТЕЛЬНО использует маску ложных связей.
    ''','Два компонента по ячейкам 33–34. Это не полный SW-MSA: только roll без mask и обратной сборки не реализует Swin.',[33,34,35])

    N('clip',r'''
    Нормированные embeddings $u_i,v_j\in\mathbb R^d$ дают logits $s_{ij}=t\,u_i^\top v_j$, $t>0$.
    Для батча из $N$ согласованных пар:
    $$\mathcal L_{I\to T}=-\frac1N\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}},\qquad
    \mathcal L_{T\to I}=-\frac1N\sum_j\log\frac{e^{s_{jj}}}{\sum_i e^{s_{ij}}},\quad
    \mathcal L=\tfrac12(\mathcal L_{I\to T}+\mathcal L_{T\to I}).$$
    Diagonal targets — соглашение порядка примеров. В матрице $N$ positives и $N^2-N$ negatives;
    эквивалентные подписи вне диагонали могут стать ложными negatives.
    Для нормированных векторов dot product равен cosine. Общий положительный scale меняет softmax, но не argmax.
    ''','''
    import torch
    import torch.nn.functional as F

    def clip_loss(image_features, text_features, log_scale):
        u = F.normalize(image_features, dim=-1)
        v = F.normalize(text_features, dim=-1)
        assert u.shape == v.shape
        logits = log_scale.exp() * (u @ v.T)
        targets = torch.arange(len(u), device=u.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    # loss = clip_loss(projected_images, projected_texts, log_scale)
    ''','Схема обучения сопоставлению по лекции. Encoder и projection heads предполагаются снаружи; log_scale — логарифм положительного scale.',[36,39,40])

    N('coca',r'''
    CoCa объединяет contrastive и conditional captioning objectives:
    $$\mathcal L=\lambda_{\rm con}\mathcal L_{\rm con}+\lambda_{\rm cap}\mathcal L_{\rm cap},\qquad
    \mathcal L_{\rm cap}=-\sum_t\log p_\theta(y_t\mid y_{<t},I).$$
    Нижний unimodal текстовый путь не использует image cross-attention и даёт независимый text embedding;
    верхний multimodal путь читает visual memory. В cross-attention:
    $$Q\in\mathbb R^{T\times d_k},\quad K\in\mathbb R^{N\times d_k},\quad
    V\in\mathbb R^{N\times d_v},\qquad \#\mathrm{scores}=TN.$$
    При $T=20,N=196$ получается $3920$ элементов на голову. Значения коэффициентов и sum/mean-нормировка
    должны быть заданы явно: они определяют относительный вес целей.
    ''','''
    import torch.nn.functional as F

    def coca_objective(contrastive_loss, caption_logits, target_tokens,
                       lambda_con=1.0, lambda_cap=2.0):
        # Caption logits уже выровнены с next-token targets [B,T].
        cap = F.cross_entropy(caption_logits.flatten(0, 1),
                              target_tokens.flatten(), ignore_index=-100)
        return lambda_con * contrastive_loss + lambda_cap * cap
    # contrastive_loss получают из независимых image/text embeddings.
    ''','Композиция двух loss по ячейке 41, не полная CoCa. Нельзя брать contrastive text embedding после image-conditioned cross-attention.',[41])

    N('siglip',r'''
    Нормируем признаки; $s_{ij}=t\,u_i^\top v_j+b$, $t>0$ и $b$ обучаются.
    Пусть $z_{ij}=+1$ для положительных пар и $-1$ для отрицательных. В нормировке исходного SigLIP:
    $$\mathcal L=-\frac1N\sum_{i,j}\log\sigma(z_{ij}s_{ij})
    =\frac1N\sum_{i,j}\operatorname{softplus}(-z_{ij}s_{ij}).$$
    Эквивалентная бинарная target-разметка $y=(z+1)/2$ позволяет BCE **с logits**.
    Усреднение по $N^2$ вместо $N$ уменьшит loss в $N$ раз: это другое соглашение о масштабе.
    $$\sigma(s)=\tfrac12\iff s=0\iff u_i^\top v_j=-b/t.$$
    Нет общего softmax-знаменателя строки, однако полная матрица всё ещё содержит $N^2$ пар,
    а embeddings участвуют во многих термах и имеют связанные градиенты.
    ''','''
    import torch
    import torch.nn.functional as F

    def siglip_loss(image_features, text_features, log_scale, bias):
        u = F.normalize(image_features, dim=-1)
        v = F.normalize(text_features, dim=-1)
        assert u.shape == v.shape
        logits = log_scale.exp() * (u @ v.T) + bias
        targets = torch.eye(len(u), device=u.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / len(u)
    ''','Scale и bias сохранены; sigmoid перед BCEWithLogits не вызывается. Диагональная разметка предполагает одну согласованную пару на пример.',[42])

    N('navit',r'''
    Для изображения $i$ число токенов $N_i=H_iW_i/P^2$. Packing размещает разные длины в одном буфере;
    для независимых изображений допустимость определяется принадлежностью:
    $$B_{ab}=\mathbf1[g_a=g_b]\mathbf1[a,b\text{ реальные}],\qquad
    \#\mathrm{allowed\ pairs}=\sum_i N_i^2.$$
    Внутри каждого image-блока внимание полное, между блоками запрещено. Плотная реализация может всё равно
    выделить $(\sum_iN_i)^2$ scores: маска задаёт семантику, а не гарантирует sparse execution.
    При обычном padding до максимальной длины:
    $$N_{\rm pad}=B\max_iN_i-\sum_iN_i.$$
    Длины $100,200,400$ дают $500$ padding-токенов. Pooling и позиции также должны учитывать границы изображений.
    ''','''
    import torch

    def packed_image_mask(image_ids):
        # [T]; -1 обозначает padding
        real = image_ids >= 0
        return ((image_ids[:, None] == image_ids[None, :])
                & real[:, None] & real[None, :])

    def per_image_mean(tokens, image_ids, count):
        return torch.stack([tokens[image_ids == i].mean(0) for i in range(count)])
    # allowed = packed_image_mask(ids); pooled = per_image_mean(tokens, ids, 2)
    ''','Минимальная схема изоляции и pooling. Для padded query маска даёт полностью запрещённую строку: её нужно обрабатывать отдельно либо вообще не вычислять. Каждый реальный image должен иметь токены.',[43])

    N('naflex',r'''
    Для сетки $h\times w$ патчей стороны $P$:
    $$N=hw\leq L,\qquad H'=Ph,\quad W'=Pw.$$
    Пропорция исходного изображения задаёт желательное отношение $w/h\approx W/H$;
    целочисленность сетки может потребовать приближения. Для квадратной сетки
    $$h=w\leq\lfloor\sqrt L\rfloor,\qquad H'_{\max}=P\lfloor\sqrt L\rfloor.$$
    Если **в учебной задаче** требуется точно $w=2h$, то $2h^2\leq L$ и
    $h_{\max}=\lfloor\sqrt{L/2}\rfloor$. При $L=1000$ получаем $h=22,N=968$, остаётся $32$ позиции.
    Это вывод для заданного toy-ограничения, а не универсальный алгоритм processor.
    Бинарный поиск масштаба корректен, когда округлённые $h(s),w(s)$ не убывают по $s$.
    ''','''
    from math import isqrt

    def exact_two_to_one_grid(max_tokens, patch_size=16):
        # Учебный частный случай: w = 2*h, cap >= 2
        if max_tokens < 2:
            raise ValueError("Бюджет слишком мал для сетки 1x2")
        h = isqrt(max_tokens // 2)
        w = 2 * h
        return {"height": h * patch_size, "width": w * patch_size,
                "tokens": h * w}
    # exact_two_to_one_grid(1000) -> height=352, width=704, tokens=968
    ''','Частный расчёт бюджета по мотивам ячеек 43–45; это не замена Siglip2ImageProcessorFast и не копия его правил resize.',[43,44,45])

    # The remaining lecture topics use the same mathematical/code presentation.
    add_image_notes(N)
    add_classical_notes(N)
    add_cnn_notes(N)
    add_detection_notes(N)
    add_calculations(questions)
    assert all(t.get('mathNotes') and t.get('implementation') for t in topics)


def add_calculations(questions):
    # Explicit worked equations; text answers and grading values remain unchanged.
    equations={
        'blend-72':r'0.25\cdot200+0.75\cdot100=125',
        'sift-76':r'4\cdot4\cdot8=128',
        'orb-79':r'10110\oplus10011=00101,\quad\operatorname{popcount}(00101)=2',
        'ratio-80':r'\frac{d_1}{d_2}=\frac{24}{80}=0.3',
        'pnp-82':r'X_c=X_w+3=0\quad\Longrightarrow\quad C_w=-3',
        'kalman-83':r'K=\frac{P^-}{P^-+R}=\frac9{9+1}=0.9',
        'conv-86':r'k^2C_{\rm in}C_{\rm out}=3^2\cdot32\cdot64=18432',
        'conv-87':r'3^2\cdot32+32\cdot64=288+2048=2336',
        'dilated-88':r'k_{\rm eff}=d(k-1)+1=3(3-1)+1=7',
        'normalization-90':r'N\cdot G=16\cdot8=128\quad\text{пар статистик}',
        'dropout-91':r'\frac6{1-0.25}=8',
        'metrics-94':r'P=\frac{TP}{TP+FP}=\frac8{8+2}=0.8',
        'focal-96':r'(1-p_t)^\gamma=(1-0.9)^2=0.01',
        'vit-103':r'N+1=(224/16)^2+1=197',
        'vit-104':r'N\mapsto4N\quad\Longrightarrow\quad\frac{(4N)^2}{N^2}=16',
        'iou-109':r'\operatorname{IoU}=\frac{50}{100+100-50}=\frac13',
        'nms-111':r'\operatorname{IoU}(A,B)=0.8>0.5\quad\Longrightarrow\quad\#\mathrm{keep}=1',
        'boxes-112':r'w=x_2-x_1=50-10=40',
        'yolo-113':r'5B+C=5\cdot2+20=30',
        'fcos-115':r'c=\sqrt{\frac44\frac33}=1',
    }
    def B(bank, rows):
        for row in rows.strip().splitlines():
            key,tex=row.strip().split(':',1)
            equations[f'tx-{bank}-{key}']=tex.strip()
    B('bert',r'''
    selected-count:2000\cdot0.15=300
    mask-count:300\cdot0.8=240
    random-share:0.15\cdot0.10\cdot100\%=1.5\%
    score-elements:T^2=128^2=16384
    head-dim:d/h=768/12=64
    ''')
    B('vit',r'''
    patch-count:N=(256/16)(384/16)=384
    patch-vector:P^2C=8^2\cdot3=192
    patch-params:P^2Cd=16^2\cdot3\cdot256=196608
    position-params:(N+1)d=197\cdot768=151296
    pair-count:(64+1)^2=4225
    ''')
    B('swin',r'''
    window-tokens:M^2=7^2=49
    window-score:M^4=7^4=2401
    window-count:(56/7)^2=64
    total-pairs:64\cdot49^2=153664
    relative-table:(2M-1)^2=13^2=169
    relative-heads:6\cdot13^2=1014
    merged-count:(56/2)^2=784
    shift-units:s=\lfloor7/2\rfloor=3
    ''')
    B('clip',r'''
    pair-matrix:N^2=8^2=64
    negatives:N^2-N=64-8=56
    projection:768\cdot256=196608
    cosine:\cos\theta=\frac{3\cdot0+4\cdot5}{\sqrt{3^2+4^2}\sqrt{0^2+5^2}}=0.8
    symmetric-loss:(0.4+0.6)/2=0.5
    ''')
    B('coca',r'''
    weighted-loss:1\cdot0.5+2\cdot1.5=3.5
    cross-pairs:TN=20\cdot196=3920
    two-heads:hTN=8\cdot20\cdot196=31360
    projection-head:768\cdot512=393216
    unimodal-causal:T(T+1)/2=10\cdot11/2=55
    ''')
    B('siglip',r'''
    positive-fraction:\frac{N}{N^2}=\frac1{20}=0.05
    negative-pairs:16^2-16=240
    scaled-logit:s=10\cdot0.3-2=1
    threshold-cos:s=0\quad\Longrightarrow\quad\cos\theta=-b/t=6/10=0.6
    block-count:(1000/250)^2=16
    ''')
    B('navit',r'''
    rect-patches:(224/16)(448/16)=392
    two-lengths:196+392=588
    fill-padding:1024-(392+392+196)=44
    ordinary-pad:3\cdot400-(100+200+400)=500
    block-pairs:4^2+3^2=25
    forbidden-cross:4\cdot3+3\cdot4=24
    max-fitting:\lfloor1000/196\rfloor=5
    area-scaling:\frac{(3H)(3W)}{HW}=9
    block-ratio:\frac{200^2}{100^2+100^2}=2
    normalized-use:800/1000=0.8
    ''')
    B('naflex',r'''
    budget-square:P\lfloor\sqrt L\rfloor=16\cdot32=512
    ratio-two:2h^2\leq512\quad\Longrightarrow\quad h_{\max}=16
    ratio-two-width:W'=Pw=16\cdot32=512
    unused-cap:1000-22\cdot44=32
    ceil-example:\lceil31/16\rceil\lceil47/16\rceil=2\cdot3=6
    embedding-projection:768\cdot1152=884736
    fill-batch:512-384=128
    attention-factor:\frac{1024^2}{256^2}=16
    max-width-fixed-height:w_{\max}=\lfloor600/20\rfloor=30
    area-from-tokens:NP^2=300\cdot8^2=19200
    ''')
    B('moe',r'''
    expert-total:8\cdot2=16\quad\text{млн}
    active-params:2\cdot3=6\quad\text{млн}
    relative-all:N/k=16/2=8
    assignments:1000\cdot2=2000
    balanced-load:2000/8=250
    router-weights:dN=256\cdot8=2048
    expert-ffn:2dd_{\rm ff}=2\cdot128\cdot512=131072
    mixture-output:0.75\cdot2+0.25\cdot6=3
    topk-factor:4/1=4
    active-fraction:k/N=2/32=0.0625
    ''')
    B('detr',r'''
    empty-slots:Q-M=100-7=93
    cross-count:QN=100\cdot196=19600
    query-self-count:Q^2=100^2=10000
    embedding-params:Qd=100\cdot256=25600
    hungarian-toy:\min(1+100,\ 2+1.1)=\min(101,3.1)=3.1
    ''')
    B('transformer',r'''
    scale-value:\sqrt{d_k}=\sqrt{64}=8
    allowed-triangle:T(T+1)/2=8\cdot9/2=36
    pe-size:512\cdot768=393216
    score-size:nm=12\cdot20=240
    ffn-size:2dd_{\rm ff}=2\cdot64\cdot256=32768
    ''')
    numeric={q['id'] for q in questions if q['type']=='number'}
    assert numeric==set(equations),(numeric-set(equations),set(equations)-numeric)
    for q in questions:
        if q['id'] in equations:q['calculation']=equations[q['id']]


def add_image_notes(N):
    N('raster',r'''
    Растр $I\in\mathbb R^{H\times W\times C}$. Перестановка RGB/BGR меняет порядок компонент,
    а переход к grayscale уменьшает число каналов. В принятой в лекции аппроксимации:
    $$Y'=0.299R'+0.587G'+0.114B'.$$
    Штрих напоминает, что это коэффициенты для кодированных компонент; физическую luminance линейного RGB
    и другие стандарты коэффициентов не следует смешивать. Для красного $(255,0,0)$ получаем $Y'\approx76.245$.
    ''','''
    import numpy as np

    def bgr_to_gray(bgr):
        rgb = bgr[..., ::-1].astype(np.float32)
        return np.clip(rgb @ np.array([0.299, 0.587, 0.114]), 0, 255).astype(np.uint8)
    # gray = bgr_to_gray(image_bgr)  # [H,W,3] -> [H,W]
    ''','Формула лекции и явная перестановка каналов. Округление может отличаться от оптимизированного cv2.cvtColor на один уровень.',[6,7,8])
    N('brightness',r'''
    $$I'=\operatorname{clip}((I-128)C+128+\Delta B,0,255),\qquad
    y=x^{1/\gamma},\quad x\in[0,1],\ \gamma>0.$$
    При этой записи $\gamma>1$ осветляет значения внутри $(0,1)$. При записи $y=x^\gamma$
    направление противоположное. Float-преобразование должно предшествовать арифметике:
    $250+20$ после clip даёт $255$, а в uint8 возможно переполнение до $14$.
    Степенная кривая — аппроксимация; точная передаточная функция sRGB кусочная.
    ''','''
    import numpy as np

    def adjust(image, contrast=1.0, brightness=0.0, gamma=1.0):
        if gamma <= 0: raise ValueError("gamma > 0")
        x = np.clip((image.astype(np.float32)-128)*contrast + 128 + brightness, 0, 255)
        return np.clip(255 * (x / 255) ** (1 / gamma), 0, 255).astype(np.uint8)
    # brighter = adjust(image, brightness=20, gamma=1.2)
    ''','Порядок операций указан явно; перестановка gamma и линейной коррекции даст другое преобразование.',[21,22,25,26])
    N('histogram',r'''
    Для $n$ пикселей и $L$ уровней:
    $$h(k)=\sum_{x,y}\mathbf1[I(x,y)=k],\quad p(k)=h(k)/n,\quad
    F(k)=\sum_{j=0}^k p(j),\quad T(k)\approx(L-1)F(k).$$
    CDF не убывает, поэтому LUT сохраняет порядок интенсивностей. В практической реализации часто вычитают
    минимальное ненулевое CDF, чтобы полезный диапазон начинался с нуля. Для константного изображения
    знаменатель такой нормировки нулевой — его обрабатывают отдельно. CLAHE ограничивает локальные гистограммы,
    поэтому не тождественна одной глобальной LUT.
    ''','''
    import cv2

    def equalize_gray(gray):  # uint8, [H,W]
        global_result = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return global_result, clahe.apply(gray)
    ''','Два вызова по лекции/семинару. Для цветного изображения обычно корректируют выбранный канал яркости; независимое equalize RGB может менять цвета.',[23,24])
    N('whitebalance',r'''
    В предположении «серого мира» средние каналов должны быть близки. Один вариант gains:
    $$\mu_c=\frac1{HW}\sum_{x,y}I_c(x,y),\quad \mu=\frac13\sum_c\mu_c,\quad
    g_c=\frac{\mu}{\mu_c},\quad I'_c=\operatorname{clip}(g_c I_c,0,255).$$
    После умножения среднее без clip стало бы $\mu$ в каждом канале. Clip и отсутствие нужных цветов
    нарушают это рассуждение. Если $\mu_c=0$, по такому изображению нельзя восстановить отсутствующий канал.
    ''','''
    import numpy as np

    def gray_world(image):
        x = image.astype(np.float32)
        means = x.mean(axis=(0, 1))
        target = means.mean()
        gains = np.divide(target, means, out=np.ones_like(means), where=means > 0)
        return np.clip(x * gains, 0, 255).astype(np.uint8)
    ''','Симметричный вариант серого мира. Оставление нулевого канала неизменным — явное правило обработки вырожденного случая.',[27,28])
    N('threshold',r'''
    $$B(x,y)=\mathbf1[I(x,y)>T],\qquad
    T^*=\arg\max_T\omega_0(T)\omega_1(T)(\mu_0(T)-\mu_1(T))^2.$$
    В Отсу веса $\omega$ — доли пикселей классов, $\mu$ — их средние. Максимизация межклассовой дисперсии
    эквивалентна минимизации внутриклассовой при фиксированной общей дисперсии. Пустые классы исключаются.
    В адаптивном пороге $T(x,y)=\sum_{u,v}w_{uv}I(x+u,y+v)-C$, где веса нормированы:
    локальная оценка помогает при неоднородном освещении, но зависит от масштаба окна.
    ''','''
    import cv2

    def threshold_variants(gray):
        otsu_T, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, blockSize=11, C=2)
        return otsu_T, otsu, adaptive
    ''','Вход — одноканальный uint8. blockSize нечётный и больше единицы; приведённый размер — пример, а не универсальный выбор.',[29,30,31,33,34])
    N('blend',r'''
    $$I=\alpha F+(1-\alpha)B,\qquad 0\leq\alpha\leq1.$$
    Коэффициенты суммируются в единицу: при $F=B$ результат равен этому же изображению.
    При $\alpha=0$ виден фон, при $\alpha=1$ — foreground; при $F=200,B=100,\alpha=0.25$ получаем $125$.
    Для пространственной маски $\alpha(x,y)$ те же операции выполняются по пикселям и всем каналам.
    Это простая схема blending лекции; она не описывает полностью композитинг двух произвольных RGBA-слоёв.
    ''','''
    import numpy as np

    def alpha_blend(foreground, background, alpha):
        a = np.asarray(alpha, dtype=np.float32)
        if a.ndim == 2: a = a[..., None]
        if np.any((a < 0) | (a > 1)): raise ValueError("alpha вне [0,1]")
        mixed = a * foreground.astype(np.float32) + (1-a) * background.astype(np.float32)
        return np.clip(mixed, 0, 255).astype(np.uint8)
    ''','Foreground и background имеют одинаковую форму. Для физически корректного смешивания света учитывают линеаризацию; здесь воспроизводится простая схема лекции.',[37,38])
    N('blur',r'''
    Для линейного сглаживания ядро обычно нормировано:
    $$I'(x,y)=\sum_{u,v}K(u,v)I(x+u,y+v),\quad \sum_{u,v}K(u,v)=1,\quad
    K_G(u,v)\propto e^{-(u^2+v^2)/(2\sigma^2)}.$$
    Поэтому константная область остаётся константной вдали от влияния границ.
    У среднего $k\times k$ веса равны $1/k^2$; у Гаусса ближние соседи весят больше.
    Медианный фильтр берёт медиану окна и **не** является свёрткой: нелинейную медиану нельзя вынести через сумму.
    ''','''
    import cv2

    def smooth_variants(image):
        mean = cv2.blur(image, (5, 5))
        gaussian = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
        median = cv2.medianBlur(image, 5)
        return mean, gaussian, median
    ''','Примеры пространственных фильтров семинара. Размер окна, sigma и обработка границ — параметры задачи, не взаимозаменяемые константы.')
    N('edges',r'''
    $$G_x=K_x\star I,\quad G_y=K_y\star I,\quad
    |G|=\sqrt{G_x^2+G_y^2},\quad \theta=\operatorname{atan2}(G_y,G_x),\quad
    \nabla^2I=\frac{\partial^2I}{\partial x^2}+\frac{\partial^2I}{\partial y^2}.$$
    $\star$ здесь обозначает cross-correlation: именно её реализует cv2.filter2D без переворота ядра.
    Для симметричного ядра она совпадает со свёрткой; для антисимметричного Sobel знак важен.
    Производная константной области равна нулю. Canny дополняет градиент сглаживанием, подавлением немаксимумов
    и связью слабых границ с сильными через два порога.
    ''','''
    import cv2
    import numpy as np

    def gradients(gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.hypot(gx, gy), np.arctan2(gy, gx)
    # magnitude, direction = gradients(gray)  # float сохраняет знак производной
    ''','Промежуточные градиенты нельзя бездумно хранить в uint8: отрицательные значения будут потеряны.',[39,40,41,42,47])
    N('morphology',r'''
    Для бинарного множества foreground $A$ и структурного элемента $S$:
    $$A\ominus S=\{x:x+S\subseteq A\},\quad A\oplus S=\{a+s:a\in A,s\in S\},$$
    $$A\circ S=(A\ominus S)\oplus S,\qquad A\bullet S=(A\oplus S)\ominus S.$$
    Opening удаляет структуры, в которые не помещается S; closing соединяет небольшие разрывы и закрывает отверстия
    соответствующего масштаба. Порядок операций важен. Для произвольных несимметричных S согласовывают отражение
    и anchor; обычное квадратное симметричное ядро снимает эту неоднозначность.
    ''','''
    import cv2
    import numpy as np

    def clean_binary(mask):
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return opened, closed
    ''','В бинарной маске 0 — фон, 255 — foreground. Opening и closing возвращаются отдельно, чтобы сравнить их эффекты.',[45,46])
    N('fourier',r'''
    $$F(u,v)=\sum_{x=0}^{H-1}\sum_{y=0}^{W-1}I(x,y)e^{-2\pi i(ux/H+vy/W)},\quad
    I'=\mathcal F^{-1}(MF).$$
    Умножение спектра на частотную маску соответствует пространственной фильтрации.
    Резкий срез частот даёт колеблющийся пространственный отклик — ringing у контрастных границ.
    Плавная маска уменьшает резкость перехода, но меняет компромисс детализация/сглаживание.
    fftshift нужен для удобного центрирования низких частот; перед обратным преобразованием его отменяют.
    ''','''
    import numpy as np

    def lowpass_fft(gray, sigma=15.0):
        if sigma <= 0: raise ValueError("sigma > 0")
        H, W = gray.shape
        y, x = np.ogrid[:H, :W]
        mask = np.exp(-((y-H//2)**2 + (x-W//2)**2) / (2*sigma**2))
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        return np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    ''','sigma измеряется в индексах частотной сетки. FFT без дополнения подразумевает периодические границы; результат оставлен float.',[49,50,51,52])
    N('geometry',r'''
    Аффинное преобразование сохраняет параллельность, перспективное задаётся однородными координатами:
    $$\begin{pmatrix}wx'\\wy'\\w\end{pmatrix}=H\begin{pmatrix}x\\y\\1\end{pmatrix},\qquad
    x'=\frac{h_{11}x+h_{12}y+h_{13}}{h_{31}x+h_{32}y+h_{33}},\quad
    y'=\frac{h_{21}x+h_{22}y+h_{23}}{h_{31}x+h_{32}y+h_{33}}.$$
    Масштаб всей H не меняет отображение. При обратном sampling для каждого выходного пикселя ищут координату
    во входе; bilinear использует четыре соседа с весами, суммирующимися в единицу.
    При $w=0$ конечная декартова координата не определена.
    ''','''
    import cv2
    import numpy as np

    def warp(image, homography, size):
        H = np.asarray(homography, dtype=np.float64)
        return cv2.warpPerspective(image, H, size, flags=cv2.INTER_LINEAR)
    # result = warp(image, H, (output_width, output_height))
    ''','Здесь H задаёт source→destination, а OpenCV выполняет нужное обратное sampling. size передаётся как (ширина, высота), не как shape массива.',[53,54,55,56,57])
    N('matching',r'''
    Для окна изображения $X$ и шаблона $T$ одинаковой формы:
    $$\operatorname{MSE}(X,T)=\frac1n\sum_i(X_i-T_i)^2,$$
    $$\operatorname{ZNCC}(X,T)=\frac{\sum_i(X_i-\bar X)(T_i-\bar T)}
    {\sqrt{\sum_i(X_i-\bar X)^2\sum_i(T_i-\bar T)^2}}.$$
    MSE минимизируют, ZNCC максимизируют. Вычитание средних снимает общий яркостный сдвиг;
    нормировка снимает положительный scale при ненулевой дисперсии. У константного шаблона знаменатель нулевой.
    Обозначение NCC иногда используют для разных вариантов; здесь явно записан вариант с центрированием.
    ''','''
    import cv2

    def locate_template(image, template):
        if template.std() == 0: raise ValueError("Нужен неконстантный шаблон")
        scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, best, _, top_left = cv2.minMaxLoc(scores)
        return top_left, best
    ''','Пример для grayscale. Карта scores имеет размер (H-h+1,W-w+1); это положение левого верхнего угла шаблона.',[58,59,60])
    N('augmentation',r'''
    Для posterize с $b$ сохранёнными битами из 8:
    $$q=2^{8-b},\qquad I'=q\left\lfloor I/q\right\rfloor.$$
    Уровней не больше $2^b$: при $b=4$ шаг $16$, и $255\mapsto240$.
    Для геометрической аугментации преобразование $T$ применяют согласованно к данным и геометрической разметке:
    $$I'=T(I),\qquad y'=T_{\rm labels}(y).$$
    У bbox после поворота преобразуют углы, затем при необходимости берут axis-aligned оболочку и clip.
    Цветовая аугментация обычно не меняет координаты рамок. Преобразование должно сохранять смысл целевой задачи.
    ''','''
    import numpy as np

    def posterize(image, bits=4):
        if not 1 <= bits <= 8: raise ValueError("bits от 1 до 8")
        step = 2 ** (8 - bits)
        return ((image.astype(np.uint16) // step) * step).astype(np.uint8)
    # augmented = posterize(image, bits=4)
    ''','Детерминированная часть аугментации семинара. В pipeline выбор параметров обычно случаен; здесь он зафиксирован для понимания эффекта.')


def add_classical_notes(N):
    N('harris',r'''
    Разложение яркости при малом сдвиге $\delta=(u,v)^\top$ даёт
    $$I(x+u,y+v)-I(x,y)\approx I_xu+I_yv,\quad
    E(\delta)\approx\delta^\top M\delta,\quad M=\sum_W w\begin{pmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{pmatrix}.$$
    $$R=\det M-k(\operatorname{tr}M)^2=\lambda_1\lambda_2-k(\lambda_1+\lambda_2)^2.$$
    Два больших собственных значения означают изменение яркости в обоих направлениях — угол.
    Одно большое и одно малое — ребро; оба малые — плоская область. Взвешивание и окно задают масштаб анализа.
    ''','''
    import cv2
    import numpy as np

    def harris_response(gray):
        x = gray.astype(np.float32)
        response = cv2.cornerHarris(x, blockSize=3, ksize=3, k=0.04)
        return response
    # candidates = harris_response(gray) > threshold  # затем local NMS
    ''','Порог и подавление соседних максимумов выбираются отдельно. Сама response ещё не является окончательным списком keypoints.')
    N('sift',r'''
    $$L(x,y,\sigma)=G_\sigma*I,\qquad D(x,y,\sigma)=L(x,y,k\sigma)-L(x,y,\sigma).$$
    При $k$ близком к 1 из производной по масштабу Gaussian scale-space:
    $$D\approx(k-1)\sigma\frac{\partial L}{\partial\sigma}
    =(k-1)\sigma^2\nabla^2L.$$
    Поэтому DoG приближает scale-normalized LoG. Экстремумы ищут в пространстве и соседних масштабах.
    Ориентация нормализует поворот; стандартный дескриптор $4\times4$ ячейки по $8$ bins имеет $128$ компонент.
    Это устойчивость при оговорённых изменениях, а не точная инвариантность к любому ракурсу.
    ''','''
    import cv2

    def sift_matches(gray_a, gray_b):
        sift = cv2.SIFT_create()
        ka, da = sift.detectAndCompute(gray_a, None)
        kb, db = sift.detectAndCompute(gray_b, None)
        if da is None or db is None: return ka, kb, []
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < .75*pair[1].distance]
        return ka, kb, good
    ''','SIFT использует вещественные дескрипторы и L2. Ratio .75 — пример порога; для геометрии после него нужен RANSAC.')
    N('orb',r'''
    Бит BRIEF сравнивает интенсивности пары внутри патча:
    $$\tau(p;x,y)=\mathbf1[p(x)<p(y)],\qquad
    d_H(a,b)=\sum_i\mathbf1[a_i\ne b_i]=\operatorname{popcount}(a\oplus b).$$
    ORB добавляет ориентацию и масштабную пирамиду к FAST/BRIEF-подобной схеме.
    Hamming соответствует числу разных бинарных тестов, а не евклидову расстоянию байтов, в которые они упакованы.
    Здесь предполагается обычный ORB с WTA_K=2; варианты WTA_K=3/4 требуют Hamming2.
    ''','''
    import cv2

    def orb_correspondences(gray_a, gray_b):
        orb = cv2.ORB_create(nfeatures=500, WTA_K=2)
        ka, da = orb.detectAndCompute(gray_a, None)
        kb, db = orb.detectAndCompute(gray_b, None)
        if da is None or db is None: return ka, kb, []
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
        return ka, kb, sorted(matches, key=lambda m: m.distance)
    ''','Cross-check оставляет взаимных ближайших соседей; он не гарантирует правильности геометрического соответствия.')
    N('ratio',r'''
    Для двух ближайших кандидатов с расстояниями $d_1\leq d_2$:
    $$\frac{d_1}{d_2}<\tau\quad\Longleftrightarrow\quad d_1<\tau d_2\quad(d_2>0).$$
    Критерий проверяет отличимость победителя от конкурента, а не только абсолютную близость.
    При $d_1=d_2=0$ совпадения неоднозначны — строгая форма справа их отклоняет.
    Cross-check проверяет другую симметрию: $b=\operatorname{NN}_B(a)$ и $a=\operatorname{NN}_A(b)$.
    Оба критерия работают до геометрической проверки и не заменяют её.
    ''','''
    def ratio_filter(knn_matches, threshold=0.75):
        good = []
        for pair in knn_matches:
            if len(pair) != 2: continue
            first, second = pair
            if first.distance < threshold * second.distance:
                good.append(first)
        return good
    # good = ratio_filter(matcher.knnMatch(desc_a, desc_b, k=2))
    ''','Проверка длины пары нужна: для некоторых запросов второго соседа может не оказаться.')
    N('ransac',r'''
    **Следствие процедуры RANSAC.** Если доля inliers $w$, размер минимальной выборки $s$, то вероятность
    чистой выборки примерно $w^s$. Для $K$ независимых попыток:
    $$p=1-(1-w^s)^K,\qquad K\geq\frac{\log(1-p)}{\log(1-w^s)}.$$
    Это оценка при постоянной доле inliers и независимых попытках; геометрически вырожденные выборки всё равно непригодны.
    Для гомографии минимально четыре невырожденных соответствия. Inlier определяется reprojection-error и порогом,
    а не descriptor distance. Порог зависит от единиц координат.
    ''','''
    import cv2
    import numpy as np

    def robust_homography(points_a, points_b):
        a, b = np.asarray(points_a, np.float32), np.asarray(points_b, np.float32)
        if len(a) < 4 or len(a) != len(b): raise ValueError("Нужны >=4 пары")
        H, inliers = cv2.findHomography(a, b, cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None: raise ValueError("Гомографию оценить не удалось")
        return H, inliers.ravel().astype(bool)
    ''','Порог 3 задан в пикселях входных точек. Одна H моделирует плоскость или специальные условия движения камеры; произвольная 3D-сцена может иметь параллакс.')
    N('poisson',r'''
    Математическая постановка gradient-domain blending:
    $$\min_f\int_\Omega\|\nabla f-v\|^2\,dx,\quad f|_{\partial\Omega}=g|_{\partial\Omega}
    \quad\Longrightarrow\quad \Delta f=\operatorname{div}v.$$
    При $v=\nabla s$ решение переносит градиенты источника, но границу согласует с destination.
    Поэтому это не постоянная alpha-сумма цветов. Граница области существенна: одинаковые исходные градиенты
    при другом фоне могут дать другие абсолютные цвета результата.
    ''','''
    import cv2

    def poisson_insert(source_bgr, destination_bgr, mask_u8, center):
        return cv2.seamlessClone(source_bgr, destination_bgr, mask_u8,
                                 center, cv2.NORMAL_CLONE)
    # center = (x, y) в координатах destination
    ''','Маска определяет область переноса; её непустая область должна помещаться в destination вокруг center. Код показывает вызов решателя, а не реализацию PDE.')
    N('classicaldetectors',r'''
    Для integral image $S(x,y)=\sum_{u\leq x,v\leq y}I(u,v)$ сумма прямоугольника получается четырьмя обращениями:
    $$\operatorname{sum}(R)=S(x_2,y_2)-S(x_1-1,y_2)-S(x_2,y_1-1)+S(x_1-1,y_1-1).$$
    Это позволяет быстро считать прямоугольные Haar-признаки каскада Viola–Jones.
    В HOG+SVM гистограммы направлений градиента образуют вектор $\phi(I)$, а линейный classifier
    вычисляет $s=w^\top\phi(I)+b$. Каскад, HOG-дескриптор и SVM — разные компоненты разных подходов.
    ''','''
    import cv2

    def hog_people(image_bgr):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, scores = hog.detectMultiScale(image_bgr, winStride=(8, 8), scale=1.05)
        return boxes, scores  # boxes: x,y,w,h
    ''','Готовый HOG+SVM detector из лекции. Это конкретный detector людей; установка классификатора является частью модели, а не обучением по текущей картинке.')
    N('camera',r'''
    Для pinhole-камеры без distortion:
    $$s\begin{pmatrix}u\\v\\1\end{pmatrix}
    =K[R\mid t]\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix},\quad
    K=\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}.$$
    $R,t$ переводят world→camera; после деления на глубину $Z_c$ получаем $u=f_xX_c/Z_c+c_x$,
    $v=f_yY_c/Z_c+c_y$. Radial distortion в нормированных координатах:
    $$x_d=x(1+k_1r^2+k_2r^4+k_3r^6),\quad y_d=y(1+k_1r^2+k_2r^4+k_3r^6),\quad r^2=x^2+y^2.$$
    Tangential terms добавляются отдельно. Масштаб 3D-координат и направление преобразования нужно согласовать.
    ''','''
    import cv2

    def calibrate(object_points, image_points, image_size):
        rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None)
        return rms, K, distortion, rvecs, tvecs
    # image_size=(W,H), списки точек задают несколько видов известной мишени
    ''','object_points задаются в общей системе мишени, image_points — в пикселях. Один rms не заменяет проверки распределения reprojection errors.')
    N('pnp',r'''
    При известных intrinsics PnP оценивает extrinsics из 3D–2D соответствий:
    $$R^*,t^*=\arg\min_{R\in SO(3),t}\sum_i\|x_i-\pi(K(RX_i+t))\|^2.$$
    $\pi$ включает перспективное деление, а в модели с distortion — соответствующее искажение.
    $R$ имеет три независимые степени свободы, а не девять свободных коэффициентов.
    Разные PnP-методы имеют разные требования к числу и расположению точек; неоднозначность и вырождение
    не исчезают из-за наличия вызова solvePnP.
    ''','''
    import cv2

    def estimate_pose(points3d, pixels2d, K, distortion):
        ok, rvec, tvec = cv2.solvePnP(points3d, pixels2d, K, distortion)
        if not ok: raise ValueError("PnP не сошёлся")
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec  # X_camera = R @ X_world + t
    ''','Ожидаются согласованные float-массивы точек. Полученный t — перенос world→camera, а не непосредственно положение центра камеры в world.')
    N('kalman',r'''
    Для одномерного случая лекции с $a=1$ и без управления:
    $$\hat x^-=\hat x,\quad P^-=P+Q,\quad
    K=\frac{P^-}{P^-+R},\quad\hat x^+=\hat x^-+K(z-\hat x^-),\quad P^+=(1-K)P^-.$$
    $Q$ — дисперсия process noise, $R$ — measurement noise. Gain балансирует доверие:
    большое $R$ уменьшает поправку, большое $P^-$ увеличивает её.
    При $P^-=4,R=1$ получаем $K=0.8$: измерение получает больший вес.
    Оптимальность в классической постановке предполагает линейную модель и согласованные шумовые предположения.
    ''','''
    def kalman_1d(mean, variance, measurement, process_var, measurement_var):
        if variance < 0 or process_var < 0 or measurement_var <= 0:
            raise ValueError("Дисперсии должны быть допустимыми")
        predicted_var = variance + process_var
        gain = predicted_var / (predicted_var + measurement_var)
        updated_mean = mean + gain * (measurement - mean)
        return updated_mean, (1-gain) * predicted_var, gain
    # mean, var, K = kalman_1d(10, 3, 12, 1, 1)  # 11.6, .8, .8
    ''','Скалярный фильтр для наглядности, без модели скорости и управления. Для многомерного состояния нужны матрицы динамики и наблюдения.')


def add_cnn_notes(N):
    N('conv',r'''
    Для grouped convolution без bias, с $G\mid C_{\rm in}$ и $G\mid C_{\rm out}$:
    $$P=k^2C_{\rm in}C_{\rm out}/G.$$
    Depthwise с multiplier 1 имеет $G=C_{\rm in}=C_{\rm out}$ и $k^2C_{\rm in}$ весов;
    следующая pointwise 1×1 добавляет $C_{\rm in}C_{\rm out}$:
    $$\frac{P_{\rm dw+pw}}{P_{\rm full}}=\frac1{C_{\rm out}}+\frac1{k^2}.$$
    Выигрыш по параметрам не равен автоматически выигрышу задержки. Для одной spatial-оси:
    $$H_{\rm out}=\left\lfloor\frac{H+2p-d(k-1)-1}{s}+1\right\rfloor.$$
    ''','''
    from torch import nn

    class DepthwiseSeparable(nn.Module):
        def __init__(self, cin, cout, kernel=3):
            super().__init__()
            self.dw = nn.Conv2d(cin, cin, kernel, padding=kernel//2, groups=cin, bias=False)
            self.pw = nn.Conv2d(cin, cout, 1, bias=False)

        def forward(self, x):
            return self.pw(self.dw(x))
    # layer = DepthwiseSeparable(32, 64)  # 32*9 + 32*64 весов
    ''','Линейная факторизация по лекции. Активации и нормализации намеренно опущены, чтобы сопоставить числа весов с формулой.')
    N('dilated',r'''
    Dilation увеличивает расстояние между отсчётами ядра:
    $$k_{\rm eff}=d(k-1)+1.$$
    У $3\times3$ с $d=2$ effective размер $5\times5$, но весов остаётся $9$ на пару каналов.
    Для transposed convolution spatial-форма:
    $$H_{\rm out}=(H_{\rm in}-1)s-2p+d(k-1)+\operatorname{output\_padding}+1.$$
    Transposed convolution — транспонированный линейный оператор обычной свёртки при согласованных формах,
    а не гарантированное обращение её потерь информации. output_padding уточняет форму, не добавляя измеренных деталей.
    ''','''
    from torch import nn

    def spatial_layers(channels=16):
        dilated = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        upsample = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        return dilated, upsample
    # dilated сохраняет H,W; upsample удваивает H,W
    ''','Один пример сохранения формы и один пример увеличения формы. Увеличение разрешения не означает восстановление исходного изображения.')
    N('normalization',r'''
    Общая операция:
    $$\mu=\frac1{|\mathcal A|}\sum_{i\in\mathcal A}x_i,\quad
    \sigma^2=\frac1{|\mathcal A|}\sum_{i\in\mathcal A}(x_i-\mu)^2,\quad
    y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.$$
    Различие — оси $\mathcal A$. Для NCHW BatchNorm2d считает train-статистики по N,H,W каждого канала;
    GroupNorm — по каналам группы и H,W отдельно для каждого примера; InstanceNorm — по H,W каждого sample/channel.
    LayerNorm нормирует указанные последние оси: `LayerNorm(C)` непосредственно подходит NHWC или token-вектору,
    но не означает нормирование каналов NCHW без перестановки.
    Running variance BN и forward variance имеют разные детали оценки в PyTorch; не подменяйте их одной случайной статистикой.
    ''','''
    from torch import nn

    def channel_layer_norm(x, layer_norm):
        # x NCHW -> NHWC, LN по C -> обратно NCHW
        return layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
    # y = channel_layer_norm(x, nn.LayerNorm(x.shape[1]))
    # BN: nn.BatchNorm2d(C); GN: nn.GroupNorm(4,C), C кратно 4
    ''','Пример делает ось нормировки явной. Слой LN нужно создавать в __init__ модели, а не заново на каждом train-шаге.')
    N('adaptive',r'''
    Условная нормализация меняет affine-параметры в зависимости от conditioning $c$:
    $$h=\operatorname{Norm}(x),\quad (\gamma(c),\beta(c))=f(c),\quad y=\gamma(c)\odot h+\beta(c).$$
    В residual-параметризации часто используют $(1+\gamma(c))h+\beta(c)$:
    при нулевом выходе conditioning-head она равна исходной нормировке.
    Для NCHW вектора $\gamma,\beta\in\mathbb R^{B\times C}$ расширяют до $B\times C\times1\times1$.
    Это меняет признаки на основании условия; неверное broadcasting может незаметно применить его к другой оси.
    ''','''
    from torch import nn

    class ConditionalNorm(nn.Module):
        def __init__(self, channels, condition_dim):
            super().__init__()
            self.norm = nn.GroupNorm(1, channels, affine=False)
            self.affine = nn.Linear(condition_dim, 2*channels)
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)

        def forward(self, x, condition):
            scale, shift = self.affine(condition).chunk(2, dim=-1)
            return self.norm(x) * (1+scale[:,:,None,None]) + shift[:,:,None,None]
    ''','Учебный пример conditional affine с GN; он показывает идею, но не подменяет собой все варианты AdaIN/AdaLN.')
    N('dropout',r'''
    Для inverted dropout в train:
    $$m_i\sim\operatorname{Bernoulli}(1-p),\quad \tilde x_i=\frac{m_ix_i}{1-p},\quad
    \mathbb E[\tilde x_i\mid x_i]=x_i\qquad (0\leq p<1).$$
    Компенсация $1/(1-p)$ сохраняет условное ожидание, а не каждый отдельный forward.
    В eval обычный dropout тождественен входу. DropPath применяет аналогичную идею к residual-ветви
    с общей маской на sample; это отличается от независимого зануления отдельных признаков.
    ''','''
    import torch
    from torch import nn

    class DropPath(nn.Module):
        def __init__(self, p=0.1):
            super().__init__()
            if not 0 <= p < 1: raise ValueError("0 <= p < 1")
            self.p = p

        def forward(self, x):
            if not self.training or self.p == 0: return x
            shape = (x.shape[0],) + (1,) * (x.ndim-1)
            keep = (torch.rand(shape, device=x.device) >= self.p).to(x.dtype)
            return x * keep / (1-self.p)
    # y = x + drop_path(residual_branch(x))
    ''','Маска общая для всех координат одной ветви sample. model.eval() влияет на слой, а torch.no_grad() само по себе не отключает dropout.')
    N('init',r'''
    При независимых входах и весах $z=\sum_{i=1}^{n_{\rm in}}w_ix_i$:
    $$\operatorname{Var}(z)\approx n_{\rm in}\operatorname{Var}(w)\operatorname{Var}(x).$$
    Это мотивирует выбор масштаба весов. Xavier uniform:
    $$w\sim U[-a,a],\quad a=\sqrt{6/(n_{\rm in}+n_{\rm out})}.$$
    Для ReLU He normal использует $\operatorname{Var}(w)=2/n_{\rm in}$, компенсируя потерю части второго момента
    при обнулении отрицательной половины. Это рассуждение с допущениями о распределениях, а не гарантия устойчивости любой сети.
    Нулевая инициализация всех скрытых нейронов не разрушает их симметрию.
    ''','''
    from torch import nn

    def initialize_relu_conv(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None: nn.init.zeros_(module.bias)
    # model.apply(initialize_relu_conv)
    ''','Инициализация должна соответствовать активации и направлению fan. Нули для bias не равнозначны нулевым весам всех нейронов.')
    N('resnet',r'''
    $$y=x+F(x),\qquad \frac{\partial y}{\partial x}=I+J_F(x).$$
    Тождественный путь даёт прямое слагаемое в производной, но не доказывает отсутствие vanishing/exploding gradients
    во всей сети. При изменении размерности shortcut заменяют проекцией $P(x)$ и складывают совместимые формы.
    DenseNet использует другой оператор:
    $$x_l=H_l([x_0,x_1,\ldots,x_{l-1}]),$$
    где конкатенация увеличивает число входных каналов; это не residual-сумма.
    ''','''
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.branch = nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(),
                nn.Conv2d(channels,channels,3,padding=1,bias=False), nn.BatchNorm2d(channels))

        def forward(self, x):
            return (x + self.branch(x)).relu()
    ''','Блок при неизменных формах, по принципу ResNet из лекции. Downsampling и projection shortcut здесь не показаны.')
    N('mobile',r'''
    Depthwise + pointwise экономит параметры: $k^2C_{\rm in}+C_{\rm in}C_{\rm out}$ вместо
    $k^2C_{\rm in}C_{\rm out}$. SE пересчитывает веса каналов:
    $$z_c=\frac1{HW}\sum_{h,w}x_{c,h,w},\quad s=\sigma(W_2\phi(W_1z)),\quad y_{c,h,w}=s_cx_{c,h,w}.$$
    Pooling даёт глобальный контекст, sigmoid — отдельный gate каждого канала. Это не softmax по каналам:
    сумма gates не обязана равняться единице. EfficientNet согласованно масштабирует глубину, ширину и разрешение;
    одни параметры не описывают всю стоимость исполнения.
    ''','''
    from torch import nn

    class SqueezeExcite(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            hidden = max(1, channels // reduction)
            self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels,hidden,1), nn.ReLU(),
                nn.Conv2d(hidden,channels,1), nn.Sigmoid())

        def forward(self, x):
            return x * self.gate(x)
    ''','Отдельный SE-модуль по лекции, не полный MobileNet/EfficientNet.')
    N('moderncnn',r'''
    Grouped convolution ResNeXt при $G$ группах имеет
    $$P_{\rm group}=k^2C_{\rm in}C_{\rm out}/G.$$
    Cardinality — число групп преобразований, а не число классов. В ConvNeXt spatial-смешивание и channel-смешивание
    можно разделить: depthwise kernel, затем pointwise MLP с расширением $r$:
    $$P_{\rm dw}=k^2C,\quad P_{\rm pw}\approx2rC^2.$$
    ResNeSt добавляет split-attention для взвешивания ветвей. Это разные способы организации блока;
    одинаковое слово «ветви» не делает grouped conv, SE и split-attention одним оператором.
    ''','''
    from torch import nn

    class ConvNeXtSketch(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.dw = nn.Conv2d(channels,channels,7,padding=3,groups=channels)
            self.norm = nn.LayerNorm(channels)
            self.mlp = nn.Sequential(nn.Linear(channels,4*channels),nn.GELU(),nn.Linear(4*channels,channels))

        def forward(self, x):
            z = self.dw(x).permute(0,2,3,1)
            return x + self.mlp(self.norm(z)).permute(0,3,1,2)
    ''','Схема главных операций ConvNeXt; layer scale, DropPath и полный stage design опущены, поэтому это не точный pretrained блок.')
    N('classification',r'''
    Для взаимоисключающих классов softmax CE по logits $z$:
    $$\mathcal L_{\rm CE}=-z_y+\log\sum_c e^{z_c}.$$
    Для независимых бинарных labels stable BCE по каждому logit:
    $$\ell(z,y)=\max(z,0)-zy+\log(1+e^{-|z|}).$$
    Эти записи избегают прямого логарифма округлённой вероятности. CE получает raw logits и индекс класса;
    BCEWithLogits — raw logits и float-targets той же формы. Дополнительный softmax/sigmoid на входе меняет objective.
    ''','''
    import torch.nn.functional as F

    def classification_losses(multiclass_logits, class_ids, multilabel_logits, binary_targets):
        ce = F.cross_entropy(multiclass_logits, class_ids)
        bce = F.binary_cross_entropy_with_logits(multilabel_logits, binary_targets.float())
        return ce, bce
    # class_ids [B] long; binary_targets [B,C]; logits [B,C]
    ''','Две разные постановки, не два обязательных loss для одной и той же головы. Reduction по умолчанию mean.')
    N('focal',r'''
    Для бинарной классификации $p_t=p$ при $y=1$ и $p_t=1-p$ при $y=0$:
    $$\operatorname{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t,\qquad \gamma\geq0.$$
    При $p_t\to1$ множитель подавляет лёгкие примеры. При $\gamma=0$ остаётся взвешенная CE.
    В отличие от удаления easy examples, их вклад плавно уменьшается.
    Регуляризация весов — другая часть цели, например $\mathcal L+\lambda\|w\|_2^2$;
    weight decay AdamW не следует автоматически отождествлять с L2-penalty внутри адаптивного optimizer.
    ''','''
    import torch
    import torch.nn.functional as F

    def binary_focal_loss(logits, targets, alpha=.25, gamma=2.0):
        y = targets.to(logits.dtype)
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = logits.sigmoid()
        pt = torch.where(y == 1, p, 1-p)
        at = torch.where(y == 1, alpha, 1-alpha)
        return (at * (1-pt).pow(gamma) * ce).mean()
    ''','Схема для targets строго 0/1. Sigmoid используется для модулирующего множителя, а CE всё равно считается устойчиво из logits.')
    N('metrics',r'''
    $$P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad
    F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}.$$
    Порог превращает scores в решения; его изменение меняет TP/FP/FN.
    Macro сначала считает метрику каждого класса и усредняет, micro сначала суммирует счётчики;
    из-за нелинейности дроби результаты в общем случае различны. ROC-AUC зависит от ранжирования scores,
    поэтому не вычисляется из одной фиксированной confusion matrix.
    ''','''
    def precision_recall_f1(tp, fp, fn):
        precision = tp / (tp+fp) if tp+fp else 0.0
        recall = tp / (tp+fn) if tp+fn else 0.0
        f1 = 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        return precision, recall, f1
    # precision_recall_f1(8, 2, 4) -> (.8, 2/3, 8/11)
    ''','Нулевые знаменатели здесь явно дают 0; библиотеки могут позволять другое zero_division-соглашение.')
    N('finetune',r'''
    Разделим параметры на backbone $\theta_b$ и голову $\theta_h$:
    $$\hat y=h_{\theta_h}(f_{\theta_b}(x)),\qquad
    \theta_h\leftarrow\theta_h-\eta_h\nabla_{\theta_h}\mathcal L.$$
    При feature extraction $\theta_b$ фиксированы; при fine-tuning они тоже обновляются, часто с отдельным learning rate.
    Это описание режима оптимизации, а не новая архитектура. Заморозка градиентов параметров и eval-режим —
    независимые решения: BN running statistics могут меняться в train даже при requires_grad=False.
    ''','''
    import torch

    def train_frozen_head(backbone, head, images, targets, optimizer, criterion):
        backbone.eval()
        head.train()
        optimizer.zero_grad()
        with torch.no_grad():
            features = backbone(images)
        loss = criterion(head(features), targets)
        loss.backward()
        optimizer.step()  # optimizer создан только по head.parameters()
        return float(loss.detach())
    ''','Учебная схема feature extraction с тензорным выходом backbone. Для fine-tuning убирают no_grad, выбирают train/eval и группы параметров осознанно.')


def add_detection_notes(N):
    N('boxes',r'''
    Для непрерывных координат xyxy:
    $$w=x_2-x_1,\quad h=y_2-y_1,\quad c_x=(x_1+x_2)/2,\quad c_y=(y_1+y_2)/2.$$
    Обратно $x_1=c_x-w/2$, $x_2=c_x+w/2$ и аналогично для y.
    Нормировка на размеры изображения: $\bar c_x=c_x/W,\bar c_y=c_y/H,\bar w=w/W,\bar h=h/H$.
    Здесь площадь $wh$ без $+1$; convention с включёнными целочисленными границами требует других соглашений.
    Score, class label и четыре координаты описывают разные части предсказания.
    ''','''
    import numpy as np

    def xyxy_to_cxcywh(boxes):
        boxes = np.asarray(boxes, dtype=np.float64)
        lo, hi = boxes[..., :2], boxes[..., 2:]
        if np.any(hi < lo): raise ValueError("Границы перепутаны")
        return np.concatenate(((lo+hi)/2, hi-lo), axis=-1)
    # xyxy_to_cxcywh([[0,0,10,20]]) -> [[5,10,10,20]]
    ''','Convention непрерывных координат фиксируется до вычисления IoU и обучения box regression.')
    N('iou',r'''
    $$\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|},\qquad
    \operatorname{GIoU}=\operatorname{IoU}-\frac{|C\setminus(A\cup B)|}{|C|}.$$
    C — минимальный охватывающий прямоугольник. Для непересекающихся boxes IoU равен нулю;
    дополнительный GIoU-член всё ещё различает их расположение. GIoU может быть отрицательным,
    поэтому loss $1-\operatorname{GIoU}$ может быть больше 1. Нулевые площади требуют явного соглашения.
    ''','''
    import numpy as np

    def iou_one_to_many(box, boxes):
        a, b = np.asarray(box, float), np.asarray(boxes, float)
        wh = np.maximum(0, np.minimum(a[2:], b[:,2:]) - np.maximum(a[:2], b[:,:2]))
        inter = wh.prod(axis=1)
        area_a = np.maximum(0, a[2:]-a[:2]).prod()
        area_b = np.maximum(0, b[:,2:]-b[:,:2]).prod(axis=1)
        union = area_a + area_b - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union>0)
    ''','Реализован IoU для xyxy без +1. При нулевом union здесь выбран результат 0; GIoU дополнительно требует площадь enclosing box.',[9,10,54,55])
    N('map',r'''
    Predictions сортируют по score, затем сопоставляют GT того же класса при фиксированном IoU threshold.
    Один GT нельзя засчитать дважды. Для первых k предсказаний:
    $$P_k=\frac{TP_k}{TP_k+FP_k},\quad R_k=\frac{TP_k}{N_{\rm GT}},\quad
    p_{\rm interp}(r)=\max_{\tilde r\geq r}p(\tilde r).$$
    AP — площадь под интерполированной PR-кривой, mAP усредняет AP по классам.
    Способ дискретизации важен: all-points integration, VOC 11 points и COCO 101 recall points различаются.
    COCO AP@[.50:.95] также усредняет несколько IoU thresholds и имеет дополнительные правила оценки.
    ''','''
    import numpy as np

    def ap_all_points(recall, precision):
        # По одному классу; recall уже не убывает, matching выполнен снаружи.
        r = np.r_[0.0, recall, 1.0]
        p = np.r_[0.0, precision, 0.0]
        p = np.maximum.accumulate(p[::-1])[::-1]
        changed = np.where(r[1:] != r[:-1])[0]
        return np.sum((r[changed+1]-r[changed]) * p[changed+1])
    ''','Учебный all-points вариант площади. Для точного COCO protocol используйте COCOeval, как в лекции; это не его сокращённая копия.',[12,13,15,16,17])
    N('nms',r'''
    В hard NMS сначала выбирается bbox с максимальным score; другие boxes подавляются при
    $$\operatorname{IoU}(b_{\max},b_i)>\tau.$$
    Затем процедура повторяется на оставшихся. При двух одинаковых boxes высокий score оставляет одну рамку,
    хотя совпадающие координаты сами не определяют её класс.
    Обычно NMS выполняют по классам, чтобы разные классы не подавляли друг друга.
    Soft-NMS вместо удаления уменьшает scores в зависимости от overlap; это другой алгоритм.
    ''','''
    import numpy as np

    def nms_single_class(boxes, scores, iou_fn, threshold=.5):
        boxes = np.asarray(boxes, float)
        order = np.argsort(-np.asarray(scores))
        keep = []
        while len(order):
            i = order[0]
            keep.append(int(i))
            rest = order[1:]
            order = rest[iou_fn(boxes[i], boxes[rest]) <= threshold]
        return keep
    # keep = nms_single_class(boxes, scores, iou_one_to_many)
    ''','Здесь одна категория, без предварительного score threshold. iou_fn передаётся явно; её реализация приведена в теме IoU.',[19,21,23])
    N('anchors',r'''
    Типичная параметризация смещения GT относительно anchor $a=(x_a,y_a,w_a,h_a)$:
    $$t_x=(x-x_a)/w_a,\quad t_y=(y-y_a)/h_a,\quad
    t_w=\log(w/w_a),\quad t_h=\log(h/h_a).$$
    Обратное преобразование:
    $$x=x_a+w_at_x,\quad y=y_a+h_at_y,\quad w=w_ae^{t_w},\quad h=h_ae^{t_h}.$$
    Логарифм переводит положительный scale в вещественный offset. Требуются положительные размеры anchors/GT;
    конкретные модели могут добавлять веса нормировки offsets. На карте H×W с A anchors получается HWA кандидатов.
    ''','''
    import torch

    def decode_offsets(anchors, offsets):
        # Оба [N,4], anchors в cx,cy,w,h; без дополнительных scale weights
        centers = anchors[:,:2] + anchors[:,2:] * offsets[:,:2]
        sizes = anchors[:,2:] * offsets[:,2:].exp()
        return torch.cat((centers-sizes/2, centers+sizes/2), dim=-1)
    # output [N,4] в xyxy; offsets=0 возвращают anchor
    ''','Базовая схема regression относительно anchor. Clipping, предельные значения log-scale и нормировка должны соответствовать реальному detector.')
    N('rcnn',r'''
    Shared feature map Fast/Faster R-CNN вычисляется один раз, в отличие от отдельных CNN-проходов R-CNN по crops.
    ROI pooling делит выбранную область на фиксированную сетку и агрегирует признаки каждого bin.
    Типичная multi-task цель:
    $$\mathcal L=\mathcal L_{\rm cls}+\lambda\mathbf1[\text{foreground}]\mathcal L_{\rm box}.$$
    Box regression обучается на назначенных foreground regions; background не имеет произвольного истинного bbox.
    Faster R-CNN добавляет RPN для предложений, но RPN objectness и финальная классификация ROI остаются разными головами.
    ''','''
    import torch.nn.functional as F

    def roi_pool_sketch(feature, box, output_size=(7,7)):
        # feature [1,C,H,W]; box целочисленный xyxy на ЭТОЙ feature map
        x1,y1,x2,y2 = map(int, box)
        if not (0 <= x1 < x2 <= feature.shape[-1] and 0 <= y1 < y2 <= feature.shape[-2]):
            raise ValueError("ROI должен быть непустым и внутри карты")
        return F.adaptive_max_pool2d(feature[:,:,y1:y2,x1:x2], output_size)
    ''','Только схема crop + fixed-size pooling. Она не воспроизводит все правила квантования исторического ROI Pool и не является ROI Align.')
    N('ssd',r'''
    SSD предсказывает offsets/classes для default boxes нескольких feature levels.
    Для одного изображения типичная цель:
    $$\mathcal L=\frac1{N_+}\left(\mathcal L_{\rm cls}^{+}+\mathcal L_{\rm cls}^{-}
    +\alpha\mathcal L_{\rm loc}^{+}\right).$$
    Hard negative mining выбирает negatives с наибольшим classification loss при ограничении примерно
    $N_-\leq3N_+$. Это отношение к positives, а не требование равенства классов.
    Случай $N_+=0$ требует отдельного правила нормировки и выбора negatives.
    ''','''
    import torch

    def ssd_classification_mining(loss_per_anchor, positive, ratio=3):
        # Один image; все неположительные anchors считаются допустимыми negatives.
        npos = int(positive.sum())
        negatives = loss_per_anchor[~positive]
        k = min(ratio*npos, negatives.numel())
        hard = negatives.topk(k).values if k else negatives[:0]
        return (loss_per_anchor[positive].sum() + hard.sum()) / max(npos, 1)
    ''','Реализован только mining классификационного loss. В этой учебной версии при отсутствии positives выбирается 0 negatives; ignored anchors и loc-loss обрабатываются снаружи.')
    N('yolo',r'''
    Версии YOLO различаются. Для anchor-based YOLOv3 центры задаются относительно ячейки $(c_x,c_y)$:
    $$b_x=s(\sigma(t_x)+c_x),\quad b_y=s(\sigma(t_y)+c_y),\quad
    b_w=p_we^{t_w},\quad b_h=p_he^{t_h}.$$
    Здесь s — stride, размеры anchor $p_w,p_h$ уже в пикселях изображения.
    Sigmoid ограничивает offset центра внутри ячейки; exp даёт положительный scale относительно anchor.
    YOLOv1 использует другую постановку, поэтому эти формулы нельзя приписывать всем YOLO.
    Objectness и class scores также не следует отождествлять с IoU во всех версиях.
    ''','''
    import torch

    def decode_yolov3(raw, grid_xy, anchor_wh, stride):
        # raw[...,4]; grid_xy[...,2] в ячейках; anchor_wh[...,2] в пикселях
        center = (raw[...,:2].sigmoid() + grid_xy) * stride
        size = raw[...,2:4].exp() * anchor_wh
        return torch.cat((center-size/2, center+size/2), dim=-1)
    ''','Только геометрический decode YOLOv3 при заданных соглашениях. Scores, multi-scale heads, обучение и NMS сюда не входят.')
    N('retinanet',r'''
    FPN объединяет lateral-признаки с upsample более грубого уровня:
    $$M_l=\operatorname{Conv}_{1\times1}(C_l)+\operatorname{Upsample}(M_{l+1}),\quad
    P_l=\operatorname{Conv}_{3\times3}(M_l).$$
    Совпадение числа каналов необходимо для суммы; spatial-size лучше задавать явно, особенно при нечётных размерах.
    На каждом уровне работают dense classification/regression heads; focal loss
    $-\alpha_t(1-p_t)^\gamma\log p_t$ уменьшает доминирование лёгких фоновых anchors.
    FPN — механизм разных масштабов признаков, focal loss — функция обучения; это разные части RetinaNet.
    ''','''
    from torch import nn
    import torch.nn.functional as F

    class TinyFPN(nn.Module):
        def __init__(self, input_channels=(32,64,128), width=32):
            super().__init__()
            self.lateral = nn.ModuleList(nn.Conv2d(c,width,1) for c in input_channels)
            self.smooth = nn.ModuleList(nn.Conv2d(width,width,3,padding=1) for _ in input_channels)

        def forward(self, features):  # от fine к coarse
            m = [conv(x) for conv,x in zip(self.lateral,features)]
            for i in range(len(m)-2, -1, -1):
                m[i] = m[i] + F.interpolate(m[i+1], size=m[i].shape[-2:], mode="nearest")
            return [conv(x) for conv,x in zip(self.smooth,m)]
    ''','Трёхуровневая схема FPN по лекции. Дополнительные P6/P7 и RetinaNet heads здесь не реализованы.',[44,46,47,48])
    N('centernet',r'''
    В CenterNet **Objects as Points** target heatmap имеет гауссовы пики центров:
    $$Y_{xyc}=\exp\left(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2}\right).$$
    При stride s центр разлагается на целую позицию $q=\lfloor c/s\rfloor$ и offset $o=c/s-q$,
    поэтому $c=s(q+o)$. Предсказанные width/height позволяют восстановить bbox вокруг центра.
    Указанная Gaussian — target при обучении, не обещание такого же вида предсказанной heatmap.
    Local-max suppression heatmap отличается от bbox IoU NMS. CornerNet использует пары углов — другую параметризацию.
    ''','''
    import torch.nn.functional as F

    def center_peaks(heatmap, threshold=.3):
        # heatmap [B,C,H,W], уже sigmoid probabilities
        local_max = F.max_pool2d(heatmap, 3, stride=1, padding=1)
        return (heatmap == local_max) & (heatmap > threshold)
    # positions = center_peaks(heatmap).nonzero()  # b,c,y,x
    ''','Это поиск локальных максимумов, не полный decoder: нужны top-k, offset и width/height. У плато равных максимумов схема может оставить несколько точек.')
    N('fcos',r'''
    Для location $(x,y)$ и расстояний до четырёх сторон $l,t,r,b$:
    $$\mathrm{bbox}=(x-l,y-t,x+r,y+b),\quad
    c=\sqrt{\frac{\min(l,r)}{\max(l,r)}\frac{\min(t,b)}{\max(t,b)}}.$$
    Для внутренней невырожденной точки $c\in[0,1]$; в центре $l=r,t=b$ даёт $c=1$.
    У границы один минимум стремится к нулю. Этот target по GT используется для centerness-ветви;
    на inference GT нет, centerness предсказывает модель. Единицы distances и stride должны быть согласованы.
    ''','''
    import torch

    def centerness_target(ltrb):
        # distances в одной системе координат; положительные точки внутри GT
        l,t,r,b = ltrb.unbind(-1)
        if (ltrb < 0).any(): raise ValueError("Точка вне bbox")
        eps = torch.finfo(ltrb.dtype).eps
        lr = torch.minimum(l,r) / torch.maximum(l,r).clamp_min(eps)
        tb = torch.minimum(t,b) / torch.maximum(t,b).clamp_min(eps)
        return (lr*tb).sqrt()
    ''','При вырожденных размерах здесь target 0. Это построение обучающего target, а не получение centerness из GT во время инференса.')
    N('detr',r'''
    DETR предсказывает фиксированный набор $\{(\hat p_q,\hat b_q)\}_{q=1}^Q$.
    Для M объектов ищется инъективное назначение GT в слоты:
    $$\sigma^*=\arg\min_{\sigma}\sum_{i=1}^M\left[c_{\rm cls}(y_i,\hat p_{\sigma(i)})
    +\lambda_1\|b_i-\hat b_{\sigma(i)}\|_1+\lambda_G(1-\operatorname{GIoU}(b_i,\hat b_{\sigma(i)}))\right].$$
    Matching cost и последующий classification loss не обязаны быть одной формулой: в исходном matching
    используют в том числе отрицательную вероятность класса. Неназначенные слоты учатся классу no-object;
    box loss получают назначенные реальные GT.
    Матрица стоимости $\begin{pmatrix}1&2\\1.1&100\end{pmatrix}$ имеет оптимум $2+1.1=3.1$,
    а не сумму независимых минимумов строк, использующих один слот дважды.
    На inference GT и Hungarian matching не нужны.
    ''','''
    from scipy.optimize import linear_sum_assignment

    def match_slots(cost_q_gt):
        # torch tensor [Q,M]; Q >= M, cost собрана из class/L1/GIoU
        if cost_q_gt.shape[0] < cost_q_gt.shape[1]:
            raise ValueError("Слотов меньше, чем GT")
        q_ids, gt_ids = linear_sum_assignment(cost_q_gt.detach().cpu().numpy())
        return q_ids, gt_ids
    # Затем loss по ПРЕЖНИМ torch-предсказаниям, индексированным q_ids/gt_ids.
    ''','Matching — дискретный шаг без градиента через решатель. Градиенты loss идут в выбранные исходные predictions; detach здесь относится только к cost для назначения.',[57,58,59])
