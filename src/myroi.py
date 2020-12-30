#! /usr/bin/python
import struct
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from math import sqrt
import numpy as np
from volumedata import VolumeData
from draw_c import draw_c
import os


class RoiView(VolumeData):
	def __init__(self, data, port=None):
		VolumeData.__init__(self, data)

		self.spacing = data.GetSpacing()
		self.dimensions = data.GetDimensions()
		self.origin = data.GetOrigin()


class RoiViewBuilder(object):
	def __init__(self, spacing=None, dimensions=None, origin=None, mark=None):
		volume = VolumeData()
		grid = volume.createGrid(spacing, dimensions, origin)
		array = volume.createIntegerArray(grid)
		ox, oy, oz = origin
		sx, sy, sz = spacing
		dx, dy, dz = dimensions

		index = 0
		for i in range(dz):
			for j in range(dy):
				for k in range(dx):
					array.SetValue(index, mark[i][j][k])
					index += 1
		grid.GetPointData().SetScalars(array)
		self.roiView = RoiView(grid)

	def getRoiView(self):
		return self.roiView


def area(c):
	a = 0
	ox, oy = c[-1]
	for x, y in c[:]:
		a += (y * ox - x * oy)
		ox, oy = x, y
	return a / 2


class MyRoi:
	def __init__(self, contours, name, resolution):
		self.paths = []
		self.z = []
		self.xmin = contours[0][0][0]
		self.xmax = self.xmin
		self.ymin = contours[0][0][1]
		self.ymax = self.ymin


		cont_for_sorted = []
		for c in contours:
			mmix = np.min(c[:,0])
			mmax = np.max(c[:,0])
			mmiy = np.min(c[:,1])
			mmay = np.max(c[:,1])
			bbarea = (mmax - mmix) * (mmay - mmiy)
			cont_for_sorted.append((c, bbarea))

		#sc = sorted(contours, key=lambda c: c[0][2]) # sorting by the z
		cont_for_sorted = sorted(cont_for_sorted, key=lambda c: (c[0][0][2], -c[1])) # sorting by the z ascending and negative area to make it descending
		sc = list(map(lambda c: c[0], cont_for_sorted))

		dz = sc[1][0][2] - sc[0][0][2]
		for i in range(len(sc)):  # reorder contours if the are not counter-clockwise
			c = sc[i]
			v12 = [0.5 * (c[0, 0] + c[1, 0]), 0.5 * (c[0, 1] + c[1, 1])]  # middle point of the first contour's segment
			dx = np.sign([c[1, 0] - c[0, 0]])
			dy = np.sign([c[1, 1] - c[0, 1]])
			leftNormal = (-resolution * dy, resolution * dx)  # [dx,dy] rotated by 90 degrees and scaled
			p = Path(c[:, 0:2])
			if p.contains_point((v12[0] + leftNormal[0], v12[1] + leftNormal[1])):
				pass  # path is oriented count-clockwise (it contains point located on the left side of v[0]->v[1]
			else:
				sc[i] = c[::-1]  # reverse

		self.contours = sc
		vol = 0
		for c in self.contours:
			self.paths.append(Path(c[:, 0:2]))
			vol = vol + area(c[:, 0:2])
			self.z.append(c[0][2])
			xmin = min(c[:, 0])
			xmax = max(c[:, 0])
			ymin = min(c[:, 1])
			ymax = max(c[:, 1])
			self.xmin = xmin if xmin < self.xmin else self.xmin
			self.xmax = xmax if xmax > self.xmax else self.xmax
			self.ymin = ymin if ymin < self.ymin else self.ymin
			self.ymax = ymax if ymax > self.ymax else self.ymax
		self.volume = vol * dz
		self.n = len(self.z)
		self.name = name
		self.count = 0

	def crossed(self, x0, y0, z0, dx, dy, dz):
		# print ' check [%g %g]x[%g %g]x[%g %g] for %s' % ( x0, x1, y0, y1, z0, z1, self.name )
		if z0 + dz < self.z[0] or z0 > self.z[self.n - 1]:
			return False
		if x0 + dx < self.xmin or x0 > self.xmax:
			return False
		if y0 + dy < self.ymin or y0 > self.ymax:
			return False
		for i in range(0, self.n):
			if z0 <= self.z[i] <= z0 + dx:
				# print 'testing path at %g' % ( self.z[i] )
				if self.paths[i].intersects_bbox(Bbox.from_bounds(x0, y0, dx, dy)):
					# print 'in %s' % ( self.name )
					return True
		# print 'not in %s' % ( self.name )
		return False

	def mark(self, xb, yb, dx, dy, kmax, jmax, imax, z, marks, sid, debug=False, ctgriddata=None):
		print('Marking %s by %d : [%g:%g]x[%g:%g]x[%g:%g]' % ( self.name, sid, self.xmin, self.xmax, self.ymin, self.ymax, self.z[0], self.z[self.n-1] ))
		fact = 0.1  # security factor
		if ctgriddata is not None:
			print("%s" % list(ctgriddata))
			(ctxb, ctyb, ctdx, ctdy, ctnx, ctny) = list(ctgriddata)
		else:
			(ctdx, ctdy) = (dx, dy)
		dmin = fact * min(dx, dy, ctdx, ctdy)
		toFill = set()
		iprev = -1 # used when the contours are coarser than the grid z-layers
		if debug:
			print('z in grid: <%g,%g>, z in ROI: <%g,%g>' % (z[0], z[-1], self.z[0], self.z[-1]))
		last_z = None
		do_mark = True
		sid_to_check = sid
		for l in range(0, self.n):
			if (last_z is None) or (last_z != self.z[l]):
				do_mark = True
				sid_to_check = 0
				sid_to_check_inverted = sid
				last_z = self.z[l]
			else:
				# if new contour have the same z then clear marks for the holes
				do_mark = False
				sid_to_check = sid
				sid_to_check_inverted = 0

			i = int((self.z[l] - z[0]) / (z[1] - z[0]))
			if debug:
				print('i=%d' % i)
			if i < 0 or i >= imax:
				if debug:
					print('%d not in <0,%d>' % (i, imax))
				continue
			vertices = self.contours[l][:, 0:2]
			# mark contour
			vp = vertices[-1]
			if debug:
				print('# vertices: %d' % len(vertices))
			for v in vertices:  # loop over segments vp-v
				n = int(sqrt((v[0] - vp[0]) ** 2 + (v[1] - vp[1]) ** 2) / dmin) + 1
				s = [v]
				for m in range(1, n):  # when length(vp-v) > dmin, add virtual vertices between vp and v
					dc = float(m) / n
					s.insert(-1, [vp[0] + dc * (v[0] - vp[0]), vp[1] + dc * (v[1] - vp[1])])
				# now s contains vv on vp-v segment, but without vp (it would be duplicated)
				# mark contour
				for vv in s:
					k = int((vv[0] - xb) / dx)
					k = k if k < kmax else kmax - 1
					j = int((vv[1] - yb) / dy)
					j = j if j < jmax else jmax - 1

					if do_mark:
						if marks[i][j][k] & sid == 0:
							if iprev >= 0 and i - iprev > 1:
								# fill skipped grid layers
								for ix in range(iprev + 1, i):
									marks[ix][j][k] += sid
							marks[i][j][k] += sid
					else:
						if marks[i][j][k] & sid == sid:
							if iprev >= 0 and i - iprev > 1:
								# fill skipped grid layers
								for ix in range(iprev + 1, i):
									marks[ix][j][k] -= sid
							marks[i][j][k] -= sid
				# prepare starting point for the next segment of the current contour
				vp = v
			# calculate seeds for filler
			# toFill will collect local seed-points (close to contour)
			toFill.clear()
			# (ks,js) will evaluate to central seed-point
			ks = 0
			js = 0
			ns = 0
			vp = vertices[-1]
			for v in vertices:  # loop over segments vp-v
				n = int(sqrt((v[0] - vp[0]) ** 2 + (v[1] - vp[1]) ** 2) / dmin) + 1
				s = [v]
				for m in range(1, n):  # when length(vp-v) > dmin, add virtual vertices between vp and v
					dc = float(m) / n
					s.insert(-1, [vp[0] + dc * (v[0] - vp[0]), vp[1] + dc * (v[1] - vp[1])])
				# now s contains vv on vp-v segment, but without vp (it would be duplicated)
				# collect seeds close to contour points
				dk = np.sign(v[0] - vp[0])
				dj = np.sign(v[1] - vp[1])
				if debug:
					print('%d points in (%g,%g)->(%g,%g) : %d,%d' % (len(s), vp[0], vp[1], v[0], v[1], dk, dj))
				for vv in s:
					k = int((vv[0] - xb) / dx)
					k = k if k < kmax else kmax - 1
					j = int((vv[1] - yb) / dy)
					j = j if j < jmax else jmax - 1
					if marks[i][j][k] & sid == sid_to_check_inverted:  # marks... & sid != 0
						if dj > 0 and k > 0 and marks[i][j][k - 1] & sid == sid_to_check and self.paths[l].contains_point(
								(xb + (k - 1) * dx, yb + j * dy + dy / 2)):
							if debug:
								print('v(%d,%d) seed at(%d,%d), marks=%d, sid=%d' % (k, j, k - 1, j, marks[i][j][k - 1], sid))
							toFill.add((k - 1, j))
						if dj < 0 and k < kmax - 1 and marks[i][j][k + 1] & sid == sid_to_check and self.paths[l].contains_point(
								(xb + (k + 1) * dx, yb + j * dy + dy / 2)):
							if debug:
								print('v(%d,%d) seed at(%d,%d), marks=%d, sid=%d' % (k, j, k + 1, j, marks[i][j][k + 1], sid))
							toFill.add((k + 1, j))
						if dk < 0 and j > 0 and marks[i][j - 1][k] & sid == sid_to_check and self.paths[l].contains_point(
								(xb + k * dx + dx / 2, yb + (j - 1) * dy)):
							if debug:
								print('v(%d,%d) seed at(%d,%d), marks=%d, sid=%d' % (k, j, k, j - 1, marks[i][j - 1][k], sid))
							toFill.add((k, j - 1))
						if dk > 0 and j < jmax - 1 and marks[i][j + 1][k] & sid == sid_to_check and self.paths[l].contains_point(
								(xb + k * dx + dx / 2, yb + (j + 1) * dy)):
							if debug:
								print('v(%d,%d) seed at(%d,%d), marks=%d, sid=%d' % (k, j, k, j + 1, marks[i][j + 1][k], sid))
							toFill.add((k, j + 1))
						ks += k
						js += j
						ns += 1
				# prepare starting point for the next segment of the current contour
				vp = v
			# search for seed inside the contour and not yet marked
			if ns > 0:
				ks = int(ks / ns)
				js = int(js / ns)
				if marks[i][js][ks] & sid == sid_to_check and self.paths[l].contains_point(
						(xb + ks * dx + 0.5 * dx, yb + js * dy + 0.5 * dy)):
					pass  # (ks,js) are ok
				elif ks > 0 and marks[i][js][ks - 1] & sid == sid_to_check and self.paths[l].contains_point(
						(xb + (ks - 1) * dx + 0.5 * dx, yb + js * dy + 0.5 * dy)):
					ks -= 1
				elif ks < kmax - 1 and marks[i][js][ks + 1] & sid == sid_to_check and self.paths[l].contains_point(
						(xb + (ks + 1) * dx + 0.5 * dx, yb + js * dy + 0.5 * dy)):
					ks += 1
				elif js > 0 and marks[i][js - 1][ks] & sid == 0 and self.paths[l].contains_point(
						(xb + ks * dx + 0.5 * dx, yb + (js - 1) * dy + 0.5 * dy)):
					js -= 1
				elif js < jmax - 1 and marks[i][js + 1][ks] & sid == sid_to_check and self.paths[l].contains_point(
						(xb + ks * dx + 0.5 * dx, yb + (js + 1) * dy + 0.5 * dy)):
					js += 1
				else:
					ks = -1  # to mark, that a central seed point has not been found
				if ks >= 0:
					if debug:
						print('central seed at(%d,%d), marks=%d, sid=%d' % (ks, js, marks[i][js][ks], sid))
					toFill.add((ks, js))
			if debug:
				print('Marked contour on level %d (grid level=%d) : %d pixels, %d seeds' % (l, i, ns, len(toFill)))
				draw_c(xb, yb, marks[i], dx, dy, toFill, self.name, l, vertices, sid, ctgriddata=ctgriddata)
			# flood of the i-th level:
			while len(toFill) > 0:
				(k, j) = toFill.pop()
				if do_mark:
					if marks[i][j][k] & sid == sid:
						continue
					if iprev >= 0 and i - iprev > 1:
						for ix in range(iprev + 1, i):
							marks[ix][j][k] += sid
					marks[i][j][k] += sid
				else:
					if marks[i][j][k] & sid == 0:
						continue
					if iprev >= 0 and i - iprev > 1:
						for ix in range(iprev + 1, i):
							marks[ix][j][k] -= sid
					marks[i][j][k] -= sid

				if k - 1 >= 0 and marks[i][j][k - 1] & sid == sid_to_check:
					toFill.add((k - 1, j))
				if k + 1 < kmax and marks[i][j][k + 1] & sid == sid_to_check:
					toFill.add((k + 1, j))
				if j - 1 >= 0 and marks[i][j - 1][k] & sid == sid_to_check:
					toFill.add((k, j - 1))
				if j + 1 < jmax and marks[i][j + 1][k] & sid == sid_to_check:
					toFill.add((k, j + 1))
			iprev = i
		self.countVoxels(marks, sid)

	def countVoxels(self, marks, sid):
		self.count = len(marks[(marks & sid) == sid])

	def removeCommonVoxels(self, marks, sid, tolerated):
		for i in range(len(marks)):
			for j in range(len(marks[0])):
				for k in range(len(marks[0][0])):
					if marks[i][j][k] & sid == sid and marks[i][j][k] - sid != tolerated:
						marks[i][j][k] -= sid
		self.countVoxels(marks, sid)

	# noinspection PyUnresolvedReferences
	def save_marks(self, fname, marks, sid):
		print("Saving marks to cache file: %s" % fname)
		interesting = (marks & sid) == sid
		bcode = struct.pack("i", -1)
		bnint = struct.pack("i", np.prod(interesting.shape))
		bsid = struct.pack("q", sid) # long long

		fout = open(fname, "wb")
		fout.write(bcode)
		fout.write(bnint)
		fout.write(bsid)
		interesting.tofile(fout)
		fout.close()

	# noinspection PyUnresolvedReferences
	def read_marks(self, fname, marks):
		res = False
		if os.path.isfile(fname):
			fin = open(fname, "rb")
			bcode = fin.read(4)
			code = struct.unpack("i", bcode)[0]

			if code < 0: # to znaczy, ze w formacie pliku uzywamy long long do sida
				bnint = fin.read(4)
				bsid = fin.read(8)
				nint = struct.unpack("i", bnint)[0]
				sid = struct.unpack("q", bsid)[0]
			else:
				bnint = bcode
				nint = struct.unpack("i", bnint)[0]
				bsid = fin.read(4)
				sid = struct.unpack("i", bsid)[0]
		
			print("nint = %d" % (nint))
			if nint == np.prod(marks.shape):
				interesting = np.fromfile(fin, np.bool_, nint)
				interesting = np.reshape(interesting, marks.shape)
				marks[interesting] += sid
				res = True
			else:
				print("ERROR! Size of markscache (%d) not equal size of marks (%d)" % (nint, np.prod(marks.shape)[0]))

			fin.close()

		return res


