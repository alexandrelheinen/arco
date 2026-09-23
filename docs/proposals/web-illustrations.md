# Imagens para a web

Jogo quieto para thumbnail, Open Graph e figuras de página. As sete
pranchas de [GALLERY.md](../GALLERY.md) continuam sendo o conjunto de
slide e artigo. Cada release publicada gera este jogo e anexa os PNG.

## O que pesa nas pranchas atuais

Cada prancha 16:9 é um cartaz. O desenho divide o quadro com:

- título em caixa alta, espaçado
- subtítulo de uma frase inteira
- legenda de cores no canto
- wordmark `ARCO` e três traços de cor
- bloco monoespaçado com números da execução
- duas faixas de scrim (topo ~34% da altura, base ~20%)
- bloom, grão e vinheta
- uma árvore de até 8 000 amostras, mais a solução, mais uma fita de
  velocidade

Isso funciona num slide visto de longe, em tela cheia. Num card de
~320 px, num preview do GitHub ou num `og:image`, o tipo vira ruído e a
árvore vira textura. A imagem tenta ser a página.

## Princípio

Uma imagem carrega um gesto. A frase, o número e o nome do algoritmo
ficam no HTML ao redor.

O mesmo gesto pode existir em dois fundos (escuro e claro). O que muda
é o chão, não o desenho.

## Três papéis

| Papel | Onde entra | Proporção | Mestre | O que cabe |
|---|---|---|---|---|
| Marca | favicon, avatar, lista | 1:1 | 512 px | três arcos, zero palavra |
| Thumbnail | README, card, Open Graph, release | 16:9 | 1600×900 | um gesto, zero tipo |
| Figura | corpo de uma página de docs | 16:9 | 1600×900 | o mesmo gesto, ainda sem tipo |

Open Graph (1200×630) é um recorte central do thumbnail, não um arquivo
desenhado à parte. O assunto mora no miolo, dentro de 80% da largura e
63% da altura, para o recorte 1.91:1 não cortar a curva.

A figura usa o mesmo arquivo do thumbnail. A legenda é um
`<figcaption>`, não pixels.

## Três conceitos

Os três partem da mesma geometria (obstáculos, árvore, curva). Mudam o
quanto a imagem insiste.

### 1. Traço

Fundo chapado, claro (`#f6f4ef`). Obstáculos como massas lisas
(`#e4dfd6`), sem borda vermelha, sem gradiente. A resposta é um traço
de ~2 px na cor do algoritmo. A estrutura (árvore ou frente de busca),
quando existe, fica num cinza-tinta a baixa opacidade.

Leitura: figura de texto técnico, no tamanho de um card. É o conceito
das páginas de documentação em fundo claro.

### 2. Sinal

Fundo chapado, escuro (`#0c1016`). Obstáculos um tom acima do fundo
(`#1a2030`), quase silhueta. A curva é a única coisa luminosa, na cor
canônica do algoritmo, sem bloom. A árvore, se entrar, fica a ~20% de
opacidade.

Leitura: o card do README e o preview de rede social, que aparecem
pequenos e em geral sobre fundo escuro. É o conceito recomendado para
thumbnail.

### 3. Textura

A árvore densa ocupa o quadro, sem título e sem métrica. O caminho
solução é o único traço claro. Funciona como hero largo (a partir de
~1200 px). A 320 px a filigrana fecha e o gesto some.

Este conceito já existe, na prática, no miolo da prancha `01_field`.
Ele permanece na galeria de slides. Não entra como thumbnail.

### Recomendação

Thumbnail e Open Graph seguem o Sinal. Figuras de página clara seguem
o Traço, com a mesma geometria. A Textura fica nas pranchas de deck.
Um assunto não ganha as duas linguagens ao mesmo tempo no mesmo lugar:
o README não mistura cartaz e thumbnail.

## O jogo

Seis gestos cobrem o que o site precisa dizer. Cada um é uma imagem.
Nenhum repete o outro.

A marca é à parte: não entra dentro das seis.

### Marca — `mark`

Três arcos curtos, concêntricos no mesmo centro, nas cores de RRT*,
SST e A*. Fundo chapado do tema. Sem palavra, sem ícone de "rota", sem
o símbolo Material que hoje está em `docs/images/arco.svg`.

```text
+------------------+
|                  |
|      ~~~         |   azul   #4477CC
|     ~~~          |   verde  #44AA66
|    ~~~           |   violeta #7744BB
|                  |
+------------------+
```

Uso: favicon, avatar do repositório, marca d'água opcional fora do
thumbnail (no layout da página, não no PNG).

### 1. `arc` — o caminho

Uma curva lisa contorna três corpos. Sem árvore. É a capa: a biblioteca
devolve um caminho.

```text
+------------------------------------------+
|                                          |
|            ______                        |
|         __/      \___        ( )         |
|       _/              \___               |
|  ( ) /                     \____    ( )  |
|                                          |
+------------------------------------------+
```

Cor da curva: azul RRT* `#4477CC`. Três obstáculos, tamanhos diferentes,
afastados o bastante para a curva ser legível a 320 px.

Uso: hero do README, Open Graph, header de release.

### 2. `branch` — a busca contínua

A mesma bacia do `arc`, com uma árvore rala por baixo e a mesma curva
por cima. A árvore explica o planejamento; a curva continua sendo o
gesto.

Orçamento: no máximo 80 segmentos desenhados. Amostras a mais podem
existir na solução; o desenho fica com os ramos que ainda se separam
a 320 px.

```text
+------------------------------------------+
|          \  |   /                        |
|           \ |  /   ______                |
|            \| / __/      \___            |
|             \/ /              \___       |
|            / \                     \____ |
|                                          |
+------------------------------------------+
```

Cor: árvore azul a 20% (Sinal) ou tinta a 35% (Traço); curva azul
sólida.

Uso: card de Planning e da página de RRT*.

### 3. `flood` — a busca discreta

Uma malha grosseira, quatro ou cinco bandas de expansão, um caminho
 curto por cima. Sem contorno numerado, sem barra de escala, sem
“expansion order” escrito na imagem.

```text
+------------------------------------------+
|  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·   |
|  ·  ·  ░░ ░░ ░░ ░░ ·  ·  ▓▓ ·  ·  ·  ·   |
|  ·  ·  ░░ ▒▒ ▒▒ ░░ ·  ·  ▓▓ ·  ·  ·  ·   |
|  ·  ·  ░░ ▒▒ ██ ▒▒ ·  ·  ▓▓ ·  ·  ·  ·   |
|  ·  o--o--o--o--o--o--o  ▓▓ ·  ·  ·  ·   |
|  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·   |
+------------------------------------------+
```

Cor: bandas em violeta A* `#7744BB`, do mais claro (cedo) ao mais
escuro (tarde). Caminho em tinta clara no Sinal, tinta escura no Traço.
Dois obstáculos no máximo, para a frente não competir com um labirinto.

Uso: card de A* e da página de Mapping quando o assunto é grade.

### 4. `pair` — dois planejadores, um mapa

Dois caminhos sobre a mesma bacia de três corpos. Sem as duas árvores
ao mesmo tempo: árvore dupla é o que faz a prancha `04_contest`
ficar cheia. Aqui cada algoritmo é só a curva que devolveu.

```text
+------------------------------------------+
|                                          |
|         _____                            |
|       _/     \___        ( )             |
|     _/           \_____                  |
|    /                   \______     ( )   |
|  ( )                                 ·   |
+------------------------------------------+
```

Azul para RRT*, verde `#44AA66` para SST. As curvas se separam o
bastante para as duas cores lerem a 320 px. Sem gráfico de comprimento
embaixo.

Uso: seção que compara RRT* e SST. A página de SST aponta para esta
imagem; ela não ganha um card próprio.

### 5. `track` — seguir o caminho

Duas curvas quase coincidentes. A de referência é fina e apagada; a
executada é a cor de destaque e se descola num trecho só, o suficiente
para se ver o erro lateral. Sem raios de lookahead, sem colorir a linha
pelo erro, sem perfil de velocidade.

```text
+------------------------------------------+
|                                          |
|     ________________________________     |
|   _/                                --   |
|  /                                 /     |
|                                          |
+------------------------------------------+
```

Referência a ~40% da cor de texto do tema. Executada em azul. Nenhum
obstáculo, ou um só, se a curva precisar de um motivo para dobrar.

Uso: card de Guidance e da página de Pure Pursuit / MPCC.

### 6. `ribbon` — a curva otimizada

A curva do `arc`, com espessura proporcional à velocidade: fina na
curva fechada, mais larga no trecho reto. Sem o gráfico de velocidade
que a prancha `05_refine` coloca ao lado.

```text
+------------------------------------------+
|                                          |
|            ______                        |
|         __/      \___                    |
|       _/              \===               |
|  ( ) /                     \====   ( )   |
|                                          |
+------------------------------------------+
```

Uma cor só (azul), opacidade constante. A espessura é o dado.

Uso: página do otimizador de trajetória.

## Orçamento visual

Vale para as seis, nos dois conceitos de página (Traço e Sinal).

| Recurso | Teto |
|---|---|
| Gestos por imagem | 1 |
| Cores de algoritmo | 1, ou 2 só no `pair` |
| Obstáculos | 3, ou 2 no `flood`, ou 1 no `track` |
| Segmentos de árvore desenhados | 80 |
| Traços enfatizados | 1 (2 no `pair` e no `track`) |
| Glifos dentro do PNG | 0 |
| Bloom, grão, vinheta, gradiente | 0 |
| Eixos, grade numérica, legenda | 0 |

A árvore de 8 000 amostras continua existindo na galeria de slides e
nos números da página. O thumbnail desenha a versão que ainda se lê
pequena. A legenda da página diz a contagem real; a imagem não diz.

## Cor

As cores de algoritmo são as de `src/arco/config/colors.yml`, as mesmas
do simulador. A galeria noturna usa um neon mais claro (`#5b93ef`,
`#3fc984`, `#a97bf2`); esse neon fica nas pranchas de deck.

| Papel | Sinal | Traço |
|---|---|---|
| Fundo | `#0c1016` | `#f6f4ef` |
| Obstáculo | `#1a2030` | `#e4dfd6` |
| Tinta | `#e7ebf2` | `#1c1e24` |
| Tinta apagada | `#8b95a8` | `#6e6a62` |
| RRT* | `#4477CC` | `#4477CC` |
| SST | `#44AA66` | `#44AA66` |
| A* | `#7744BB` | `#7744BB` |

O vermelho de obstáculo do simulador (`#D97070`) não entra neste jogo.
No card ele compete com a curva. No simulador ele continua.

## Texto

O PNG não tem título, subtítulo, wordmark, métrica nem legenda.

A página carrega o que a prancha antiga escrevia por cima do desenho:

```html
<figure>
  <img src="docs/images/web/arc-dark.png"
       alt="Uma curva lisa contorna três obstáculos.">
  <figcaption>
    Caminho devolvido por RRT* nesta bacia.
    A frase e o número ficam aqui, não na imagem.
  </figcaption>
</figure>
```

O `alt` descreve o que se vê. O nome do algoritmo fica no `figcaption`
ou no título da seção.

## Onde cada arquivo entra

| Arquivo | Superfície |
|---|---|
| `mark-dark.png`, `mark-light.png` | avatar, favicon |
| `arc-dark.png` | README, Open Graph, release |
| `arc-light.png` | capa quando a página é clara |
| `branch-*` | Planning, RRT* |
| `flood-*` | A*, grade |
| `pair-*` | comparação RRT* / SST |
| `track-*` | Guidance, rastreamento |
| `ribbon-*` | otimizador |

`*-dark` é o Sinal. `*-light` é o Traço. São doze PNG mais dois da
marca. O recorte Open Graph do `arc-dark` pode ser gerado na hora de
publicar; não é uma décima terceira composição.

Fora deste jogo, de propósito: reachability, o tríptico de crescimento
(1 200 / 3 000 / 8 000) e o zoom de poda. São argumentos de slide e de
artigo. A página aponta para a galeria se precisar deles.

## Relação com a galeria

| | Galeria (`nocturne`, `atlas`) | Este jogo |
|---|---|---|
| Uso | slide, artigo, header de release largo | card, Open Graph, figura de docs |
| Tipo dentro da imagem | título, subtítulo, legenda, métrica | nenhum |
| Amostras desenhadas | até 8 000 | o que ainda se lê a 320 px |
| Tratamento | bloom, grão, scrim | chapado |
| Mundo | bacia 160×90 cheia de corpos | 1 a 3 corpos |

As curvas continuam sendo saída de solver. O que muda é o mundo (menos
corpos), o quanto da árvore se desenha, e a ausência de cromo.

## Produção

O renderizador é `tools/render_web.py`. O mundo é um
`KDTreeOccupancy` e uma `EuclideanGrid` comuns, menores que a bacia da
galeria. A exportação não chama `Canvas.chrome`. Os dois fundos saem
da mesma geometria. A árvore desenhada para em 80 arestas, as mais
longas. Os PNG não entram no git: a release é que os carrega.

```bash
python3 tools/render_web.py --output /tmp/release_images
bash scripts/generate_release_images.sh --out-dir /tmp/release_images
```

O cache do solver fica em `tools/output/web_cache/`.

## Release

`.github/workflows/release.yml` dispara ao publicar uma release, no
mesmo evento dos vídeos do simulador. O job `generate-images` renderiza
o jogo. O job `publish-images` anexa cada arquivo com
`scripts/publish_release_images.sh`. A lista de nomes vem de
`python3 tools/render_web.py --list`, que também inclui `arc-og.png`,
o recorte 1200×630 de `arc-dark.png`.

Os arquivos numa release `vX.Y.Z`:

- `mark-dark.png`, `mark-light.png`
- `arc-dark.png`, `arc-light.png`, `arc-og.png`
- `branch-dark.png`, `branch-light.png`
- `flood-dark.png`, `flood-light.png`
- `pair-dark.png`, `pair-light.png`
- `track-dark.png`, `track-light.png`
- `ribbon-dark.png`, `ribbon-light.png`

A galeria de sete pranchas não é reestilizada por este jogo e não é
anexada por esse job.

## Regras que o render segue

1. O fundo escuro é o Sinal e o fundo claro é o Traço, com a mesma
   geometria.
2. O PNG não leva tipo.
3. O jogo são os seis gestos mais a marca.
4. A galeria de sete pranchas permanece para slide e artigo.
5. As cores de algoritmo saem de `colors.yml`.
6. `docs/images/arco.svg` não é substituído. A marca de três arcos
   viaja só como arquivo da release.
