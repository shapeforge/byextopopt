#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

unsigned int index(const int y, const int x, const int z, const int width) {
	return ((y * width + x) * 3 + z);
}

double patch_distance(
	__global double *target,
	const int y,
	const int x,
	__global double *source,
	const int sy,
	const int sx,
	const int patchWidth,
	const int patchHeight,
	const int targetWidth,
	const int sourceWidth,
	const double minDist,
	__global int *occurenceMap,
	const double const_occ,
	const double lambda_occ,
	const double esim)
{
	double distance = 0;
	for (int j = 0; j < patchHeight; ++j){
		for (int i = 0; i < patchWidth; ++i){
			int k = 1;  // only compare with first RGB channel
			double t = fabs((target[index(j + y, i + x, k, targetWidth)] - source[index(j + sy, i + sx, k, sourceWidth)]));
			distance += native_powr((float)t, (float)esim);
			if (distance >= minDist) return -1; // early termination
			if (lambda_occ > 0){
				double occMapVal = (double)((uint)occurenceMap[((j + sy) * sourceWidth + (i + sx))]);
				distance += lambda_occ * occMapVal * const_occ;
			}
		}
	}
	return distance;
}

double omega(
	__global int *occurenceMap,
	const int sy,
	const int sx,
	const int patchWidth,
	const int patchHeight,
	const int sourceWidth)
{
	double accum = 0.0;
	for (int j = 0; j < patchHeight; ++j)
		for (int i = 0; i < patchWidth; ++i)
			accum += (double)((uint)occurenceMap[((j + sy) * sourceWidth + (i + sx))]);
	return accum / (patchWidth * patchHeight);
}

int random(int start, int end,unsigned int seed) {
	unsigned int num = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	return num % (end - start) + start;
}

#define nff(i,j,k) nff[((i) * effectiveWidthTarget +(j))*3+(k)]

__kernel void build_occurrence_map(
	__global int *occurenceMap,
	__global double *nff,
	const int patchHeight,
	const int patchWidth,
	const int sourceWidth,
	const int effectiveWidthTarget)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int sy = nff(y, x, 0);
	int sx = nff(y, x, 1);
	for (int py = 0; py < patchHeight; ++py){
		for (int px = 0; px < patchWidth; ++px){
			atomic_inc(&(occurenceMap[((sy + py) * sourceWidth + (sx + px))]));
		}
	}
}

__kernel void random_fill(
	__global double *nff,
	const int effectiveHeightSource,
	const int effectiveWidthSource,
	const int effectiveWidthTarget)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	int seed = y << 16 + x;
	seed = nff(y, x, 0) = random(0, effectiveHeightSource, seed);
	nff(y, x, 1) = random(0, effectiveWidthSource, seed);
}

__kernel void initialize_distance(
	__global double *target,
	__global double *source,
	__global double *nff,
	__global int *occurenceMap,
	const int patchHeight,
	const int patchWidth,
	const int targetWidth,
	const int sourceWidth,
	const int effectiveHeightSource,
	const int effectiveWidthSource,
	const int effectiveWidthTarget,
	const double lambda_occ,
	const double esim,
	double const_occ)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	nff(y, x, 2) = patch_distance(target, y, x, source, nff(y,x,0), nff(y,x,1), patchWidth,
		patchHeight, targetWidth, sourceWidth, DBL_MAX, occurenceMap, const_occ, lambda_occ,esim);
}

#define checkBound(val, ini, end) (val >= ini && val < end)

__kernel void propagate(
	__global double *target,
	__global double *source,
	__global double *nff,
	__global int *occurenceMap,
	const int patchHeight,
	const int patchWidth,
	const int targetHeight,
	const int targetWidth,
	const int sourceHeight,
	const int sourceWidth,
	const int effectiveHeightTarget,
	const int effectiveWidthTarget,
	const int effectiveHeightSource,
	const int effectiveWidthSource,
	const int iteration,
	const double lambda_occ,
	const double esim,
	const double const_occ)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	//propagation step
	int dir = 1;
	if (iteration % 2 == 1) dir = -1;
	double currentD = nff(y, x, 2);
	if (checkBound(y - dir, 0, effectiveHeightTarget)){
		int sy = nff(y - dir, x, 0) + dir;
		int sx = nff(y - dir, x, 1);
		if (checkBound(sx, 0, effectiveWidthSource) && checkBound(sy, 0, effectiveHeightSource)){
			double topD = patch_distance(target, y, x, source, sy, sx, patchWidth, patchHeight,
				targetWidth, sourceWidth, currentD, occurenceMap, const_occ, lambda_occ,esim);
			if (topD >= 0 && topD < currentD){
				nff(y, x, 0) = sy;
				nff(y, x, 1) = sx;
				currentD = nff(y, x, 2) = topD;
			}
		}
	}
	if (checkBound(x - dir, 0, effectiveWidthTarget)){
		int sy = nff(y, x - dir, 0);
		int sx = nff(y, x - dir, 1) + dir;
		if (checkBound(sx, 0, effectiveWidthSource) && checkBound(sy, 0, effectiveHeightSource)){
			double leftD = patch_distance(target, y, x, source, sy, sx, patchWidth, patchHeight,
				targetWidth, sourceWidth, currentD, occurenceMap, const_occ, lambda_occ, esim);
			if (leftD >= 0 && leftD < currentD){
				nff(y, x,0) = sy;
				nff(y, x, 1) = sx;
				currentD = nff(y, x, 2) = leftD;
			}
		}
	}

	//random search step
	unsigned int seed = iteration * x * y * effectiveWidthSource;
	int w = effectiveWidthSource, h = effectiveHeightSource;
	double expFactor = 2.0;
	double currentOmega;
	bool omegaRest = true; //additional condition, as suggested in the paper "Self tuning texture optimization"
	if (lambda_occ > 0 && omegaRest){
		currentOmega = omega(occurenceMap, nff(y, x, 0), nff(y, x, 1), patchWidth, patchHeight, sourceWidth);
	}
	double wbest = (targetHeight * targetWidth * patchHeight * patchWidth) / (sourceHeight * sourceWidth);
	while (h > 1 && w > 1){
		int x1, y1, x2, y2;
		y1 = nff(y, x, 0) - h; //"instant sythesis by numbers" random walk
		x1 = nff(y, x, 1) - w;
		y2 = nff(y, x, 0) + h;
		x2 = nff(y, x, 1) + w;
		if (x1 < 0) x1 = 0;
		if (y1 < 0) y1 = 0;
		if (x2 >= effectiveWidthSource) x2 = effectiveWidthSource - 1;
		if (y2 >= effectiveHeightSource) y2 = effectiveHeightSource - 1;
		int sx = seed = random(x1, x2, seed);
		int sy = seed = random(y1, y2, seed);
		double newD = patch_distance(target, y, x, source, sy, sx, patchWidth, patchHeight,
			targetWidth, sourceWidth, currentD, occurenceMap, const_occ, lambda_occ, esim);
		bool omegaCond = true;
		double newOmega = 0;
		if (lambda_occ > 0 && omegaRest){
			newOmega = omega(occurenceMap, sy, sx, patchWidth, patchHeight, sourceWidth);
			omegaCond = newOmega < currentOmega || newOmega <= 2 * wbest;
		}
		if (newD >= 0 && newD < currentD && omegaCond) {
			nff(y, x, 0) = sy;
			nff(y, x, 1) = sx;
			currentD = nff(y, x, 2) = newD;
			currentOmega = newOmega;
		}
		w /= expFactor;
		h /= expFactor;
	}
}

//atomic addition with double is not available in OpenCL, we need this function
void atomic_add_global(__global double *source, const double operand) {
	union {
		long intVal;
		double doubleVal;
	} newVal;
	union {
		long intVal;
		double doubleVal;
	} prevVal;

	do {
		prevVal.doubleVal = *source;
		newVal.doubleVal = prevVal.doubleVal + operand;
	} while (atomic_cmpxchg((volatile global long *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


#define EPSILON 0.0001

__kernel void build_gradient(
	__global double *target,
	__global double *source,
	__global double *nff,
	__global double *energy,
	__global double *grad,
	const int patchHeight,
	const int patchWidth,
	const int targetWidth,
	const int sourceWidth,
	const int effectiveHeightTarget,
	const int effectiveWidthTarget,
	const int effectiveHeightSource,
	const int effectiveWidthSource,
	const double esim)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	double target_xy = target[index(y, x, 0, targetWidth)];
	double energy_xy = 0.0;
	int j = max(0, y - (patchHeight - 1));
	int end_j = min(y, effectiveHeightTarget - 1);
	while (j <= end_j){
		int i = max(0, x - (patchWidth - 1));
		int end_i = min(x, effectiveWidthTarget - 1);
		while (i <= end_i){
			// grad xy
			int sy = nff(j, i, 0);
			int sx = nff(j, i, 1);
			if (checkBound(sx, 0, effectiveWidthSource) && checkBound(sy, 0, effectiveHeightSource)){
				double source_xy = source[index(sy + (y - j), sx + (x - i), 0, sourceWidth)];
				double diff = target_xy - source_xy;
				if (esim == 2.0){
					energy_xy += diff * diff;
					grad[y * targetWidth + x] += 2.0 * diff;
				}else{
					energy_xy += powr(diff * diff + EPSILON, esim / 2.0);
					grad[y * targetWidth + x] += esim * diff * powr(diff * diff + EPSILON, esim / 2.0 - 1.0);
				}
			}
			i++;
		}
		j++;
	}
	atomic_add_global(&energy[0], energy_xy);
	return;
}
