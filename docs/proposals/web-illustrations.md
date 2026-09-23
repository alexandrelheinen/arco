# Imagens para a web

O conjunto que uma release anexa. São as sete pranchas de
[GALLERY.md](../GALLERY.md), desenhadas de novo sem o cartaz: o campo de
distância, as árvores e as curvas continuam, porque são o que explica o
algoritmo. O título, a legenda, a métrica e o gráfico ao lado saem. A
frase mora no texto da página.

Um jogo anterior, de seis gestos chapados (uma curva, uma árvore de 80
arestas, uma grade de duas bandas), foi deixado de lado. Lia-se pequeno
e não mostrava o que a biblioteca faz.

## O que sai da prancha

A prancha de slide continua existindo, com cromo. A exportação web é a
mesma geometria e a mesma saída de solver, com três cortes:

- nenhum glifo no PNG: título, subtítulo, wordmark, legenda, métrica,
  rótulo de start/goal
- nenhum gráfico lateral (comprimento da rota, perfil de velocidade,
  erro de rastreamento)
- o bloom largo sob a curva vira uma auréola curta. O traço fica; a
  lâmpada não

O gradiente do palco, a grade fina e o grão leve permanecem. São o chão
da série, não informação.

## Três papéis

| Papel | Onde entra | Proporção | Mestre |
|---|---|---|---|
| Marca | favicon, avatar | 1:1 | 512 px |
| Figura | README, card, relatório, docs | 16:9 | 1920×1080 |
| Open Graph | preview de link | 1.91:1 | 1200×630, recorte de `field-nocturne` |

A figura e o card usam o mesmo PNG. A legenda é um `<figcaption>`.

## Os dois fundos

`nocturne` é o palco escuro da galeria. `atlas` é o papel. A geometria
não muda. As cores de algoritmo são as da prancha: no noturno, o neon
da galeria; no atlas, a tinta de `colors.yml`.

## As pranchas

Cada uma continua sendo a execução real do solver, no mundo 160×90 da
galeria. O que muda é o que se desenha por cima.

| Arquivo | O que se vê | De onde vem |
|---|---|---|
| `field` | Árvore densa de RRT* colorida pelo custo, a rota, a fita de velocidade | `RRTPlanner`, pruner, optimizer |
| `wavefront` | Campo de expansão do A* com contornos de esforço igual e a rota | `AStarPlanner` |
| `growth` | A mesma semente em 1 200, 3 000 e 8 000 amostras | `RRTPlanner` |
| `contest` | A*, RRT* e SST no mesmo mapa, exploração atrás da resposta | os três |
| `refine` | Caminho cru, atalhos podados, curva otimizada por velocidade | pruner, optimizer |
| `pursuit` | Referência, raios de lookahead, trajetória colorida pelo erro | `TrackingLoop`, pure pursuit |
| `reachability` | Leque do modelo de Dubins, o que o mapa admite e o que remove | `DubinsVehicle` |
| `mark` | Três arcos, nas três cores, sem palavra | desenho |

`field-og.png` é o recorte central de `field-nocturne.png`. Não é uma
oitava composição.

## Texto

```html
<figure>
  <img src="docs/images/web/wavefront-nocturne.png"
       alt="Um campo de expansão contorna os obstáculos e uma rota o atravessa.">
  <figcaption>
    A* inunda a grade antes de escolher a rota.
    A frase e o número ficam aqui, não na imagem.
  </figcaption>
</figure>
```

## Produção

```bash
python3 tools/render_web.py --output /tmp/release_images
bash scripts/generate_release_images.sh --out-dir /tmp/release_images
```

O cache do solver é o da galeria, `tools/output/gallery_cache/`. A
primeira passada resolve (o RRT* de 8 000 amostras leva cerca de 100 s);
uma mudança de traço reusa o cache.

Os PNG não entram no git. `.github/workflows/release.yml` renderiza no
evento de release publicada e anexa cada arquivo listado por
`python3 tools/render_web.py --list`.

Arquivos numa release `vX.Y.Z`:

- `mark-nocturne.png`, `mark-atlas.png`
- `field-nocturne.png`, `field-atlas.png`, `field-og.png`
- `wavefront-nocturne.png`, `wavefront-atlas.png`
- `growth-nocturne.png`, `growth-atlas.png`
- `contest-nocturne.png`, `contest-atlas.png`
- `refine-nocturne.png`, `refine-atlas.png`
- `pursuit-nocturne.png`, `pursuit-atlas.png`
- `reachability-nocturne.png`, `reachability-atlas.png`

## Regras que o render segue

1. A exportação web chama as pranchas da galeria com `quiet=True`.
2. O PNG não leva tipo.
3. O gráfico lateral não é desenhado.
4. O bloom usa a rampa curta de `illustration.quiet`.
5. As pranchas de deck, com cromo, não mudam e não são anexadas.
6. `docs/images/arco.svg` não é substituído. A marca de três arcos
   viaja só como arquivo da release.
