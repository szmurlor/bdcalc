#! /usr/bin/python

import matplotlib.pyplot as plt


def draw_c(xb, yb, grid, dx, dy, seeds, name, level, vertices, sid, ctgriddata=None):
    # f = open( "%s%03l.gif", "w" )
    # TEST dx = dy = 1
    for j in range(0, len(grid)):
        y = yb + j * dy
        for i in range(0, len(grid[0])):
            x = xb + i * dx
            plt.gca().add_patch(
                plt.Rectangle((x, y), dx, dy, facecolor='#aaaaaa' if grid[j][i] & sid == sid else '#ffffff'))
            plt.plot([x, x, x + dx, x + dx, x], [y, y + dy, y + dy, y, y], color='b')
    for (si, sj) in seeds:
        plt.gca().add_patch(plt.Rectangle((xb + si * dx, yb + sj * dy), dx, dy, facecolor='#ff0000'))
    plt.plot(vertices[:, 0], vertices[:, 1], 'go-')
    for i in range(0, len(vertices)):
        plt.annotate('%d' % (i), (vertices[i, 0], vertices[i, 1]))
    if ctgriddata is not None:
        print(ctgriddata)
        (ctxb, ctyb, ctdx, ctdy, ctnx, ctny) = ctgriddata
        minx = min(vertices[:, 0]) - ctdx
        maxx = max(vertices[:, 0]) + ctdx
        miny = min(vertices[:, 1]) - ctdy
        maxy = max(vertices[:, 1]) + ctdy
        print("(%g,%g)x(%g,%g)" % (minx, maxx, miny, maxy))
        ctnx = int(ctnx)
        ctny = int(ctny)
        for j in range(0, ctny):
            y = ctyb + j * ctdy
            if y > miny and y < maxy:
                for i in range(0, ctnx):
                    x = ctxb + i * ctdx
                    if x > minx and x < maxx:
                        # plt.gca().add_patch( plt.Rectangle( (x,y), dx, dy, facecolor='#aaaaaa' if grid[j][i] != 0 else '#ffffff' ) )
                        plt.plot([x, x, x + ctdx, x + ctdx, x], [y, y + ctdy, y + ctdy, y, y], color='y')
    plt.show()
