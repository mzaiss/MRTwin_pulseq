#include <thrust/complex.h>
#include <math.h>

typedef thrust::complex<float> cfloat;

__device__ cfloat warp_reduce(unsigned mask, cfloat val) {
	for (int offset = 16; offset > 0; offset /= 2) {
		val += cfloat(
			__shfl_down_sync(mask, val.real(), offset),
			__shfl_down_sync(mask, val.imag(), offset)
		);
	}
	return val;
}

__device__ void block_reduce_atomic_add(unsigned mask, cfloat* target, cfloat val) {
	static __shared__ cfloat shared[32];  // enough for max. nr. of warps
	int lane = threadIdx.x % 32;
	int warp = threadIdx.x / 32;
	const int warp_count = ceilf(blockDim.x / 32.0);

	// Reduce per-warp, let one lane write the result
	val = warp_reduce(mask, val);
	if (lane == 0) {
		shared[warp] = val;
	}
	__syncthreads();  // Syncronize before reading 'shared' again

	// One warp computes sum of all warps
	if (warp == 0) {
		val = (lane < warp_count) ? shared[lane] : cfloat(0.0, 0.0);
		val = warp_reduce(mask, val);

		// Only one thread per block atomically adds the value to global memory
		if (lane == 0) {
			float* flt_target = reinterpret_cast<float*>(target);
			atomicAdd(flt_target + 0, val.real());
			atomicAdd(flt_target + 1, val.imag());
		}
	}
	__syncthreads();  // Ensure this function does not produce divergence
}


// Note:
// Measured distributions come first in every repetition.
// dists[i].x = ++ or zz
// dists[i].y = -+ or -z
// dists[i].z = z+ or +z
// dists[i].w = {0: measured +, 1: +, 2: z, 3: z0}
extern "C" {
	__global__ void execute_graph(
		// Phantom data
		int voxel_count,
		int coil_count,

		// Per-voxel phantom data
		float* T1_array, float* T2_array, float* T2dash_array,
		float* B0_array, float* B1_array,
		float3* pos_array,
		float* coil_sens_array,  // coil_count floats per voxel

		// Sequence data
		int rep_count,

		// Per-repetition sequence data
		int* event_count,
		float* pulse_angle,
		float* pulse_phase,

		// Per-event sequence data
		float* event_time,
		float3* gradm_event,

		// Per-repetition graph data
		int4* dists,
		int* dist_count,

		// Per-event, per-distribution data
		cfloat* attenuation,  // sinc(k) * adc
		float* dephasing_time,  // for T2' dephasing (b.c. it's per-voxel)

		// Output, coil_count cfloats per event
		cfloat* signal
	) {
		const int voxel = blockIdx.x*blockDim.x + threadIdx.x;
		if (voxel >= voxel_count)
			return;  // Mask out threads
		const unsigned int mask = __activemask();
		
		// Constant data
		const float R1 = 1.0 / T1_array[voxel];
		const float R2 = 1.0 / T2_array[voxel];
		const float T2dash = T2dash_array[voxel];
		const float B0 = B0_array[voxel];
		const float B1 = B1_array[voxel];
		const float3 pos = pos_array[voxel];
		const float* coil_sens = coil_sens_array + voxel*coil_count;

		// Buffers for the distributions' magnetisation
		cfloat prev_dist_mag[MAX_DIST_COUNT];
		cfloat dist_mag[MAX_DIST_COUNT];
		// We start off with a single relaxed z0 state
		prev_dist_mag[0] = cfloat(1, 0);

		// These pointers will be updated after each repetition
		int4* rep_dists = dists;
		float* rep_event_time = event_time;
		float3* rep_gradm_event = gradm_event;
		// These pointers will be updated after every event
		cfloat* event_signal = signal;
		cfloat* event_attenuation = attenuation;
		float* event_dephasing_time = dephasing_time;

		// we start the simulation with the first pulse (z0 before that is
		// implicitely encoded in prev_dist_mag)
		for (int r = 0; r < rep_count; r++) {
			float angle = pulse_angle[r] * B1;
			cfloat rot = cfloat(cos(pulse_phase[r]), sin(pulse_phase[r]));
			// Calculate the (needed) rot mat elements
			cfloat z_to_z = cos(angle);
			cfloat p_to_p = z_to_z * 0.5 + 0.5;
			cfloat z_to_p = cfloat(0, -0.70710678118) * sin(angle) * rot;
			cfloat p_to_z = -thrust::conj(z_to_p);
			cfloat m_to_z = -z_to_p;
			cfloat m_to_p = (1 - p_to_p) * rot * rot;

			// Calculate the magnetisation of all distributions
			for (int d = 0; d < dist_count[r]; d++) {
				int4 dist = rep_dists[d];
				cfloat& mag = dist_mag[d];

				if (dist.w < 2) {  // (measured +) or (+) magnetisiation
					mag = (dist.x < 0 ? 0 : prev_dist_mag[dist.x] * p_to_p)
						+ (dist.y < 0 ? 0 : thrust::conj(prev_dist_mag[dist.y]) * m_to_p)
						+ (dist.z < 0 ? 0 : prev_dist_mag[dist.z] * z_to_p);
				} else {  // (z) or (z0) matnetisation
					mag = (dist.x < 0 ? 0 : prev_dist_mag[dist.x] * z_to_z)
						+ (dist.y < 0 ? 0 : thrust::conj(prev_dist_mag[dist.y]) * m_to_z)
						+ (dist.z < 0 ? 0 : prev_dist_mag[dist.z] * p_to_z);
				}
			}

			// Calculate the signal for all events
			// The rotation and relaxation is identical for all + distributions
			cfloat r2_rot = 1;
			// We don't need this right now but can compute it together with r2_rot
			float r1 = 1.0;

			for (int e = 0; e < event_count[r]; e++) {
				cfloat mag = 0;
				
				float rot_angle = rep_event_time[e] * B0
								+ rep_gradm_event[e].y * pos.y
								+ rep_gradm_event[e].z * pos.z
								+ rep_gradm_event[e].x * pos.x;
				r2_rot *= cfloat(cos(rot_angle), sin(rot_angle));
				r2_rot *= expf(-rep_event_time[e] * R2);
				r1 *= expf(-rep_event_time[e] * R1);

				// Iterate over all (measured +) distributions, they are in front
				for (int d = 0; d < dist_count[r]; d++) {
					if (rep_dists[d].w != 0) {
						break;
					}
					cfloat measured_mag = dist_mag[d] * r2_rot
						* event_attenuation[d]
						// abs() is already applied to dephasing_time
						* exp(-event_dephasing_time[d] / T2dash);
					
					for (int c = 0; c < coil_count; c++) {
						cfloat coil_signal = coil_sens[c] * measured_mag;
						block_reduce_atomic_add(mask, event_signal + c, coil_signal);
					}
				}

				// Update pointers for next event
				event_signal += coil_count;
				event_attenuation += dist_count[r];
				event_dephasing_time += dist_count[r];
			}

			// Update magnetisation of distributions that were not measured
			for (int d = 0; d < dist_count[r]; d++) {
				switch (rep_dists[d].w) {
					case 0: prev_dist_mag[d] = dist_mag[d] * r2_rot; break; // (measured +)
					case 1: prev_dist_mag[d] = dist_mag[d] * r2_rot; break;  // (+)
					case 2: prev_dist_mag[d] = dist_mag[d] * r1; break;  // (z)
					case 3: prev_dist_mag[d] = dist_mag[d] * r1 + (1 - r1); break;  // (z0)
				}
			}

			// Update pointers for next repetition
			rep_dists += dist_count[r];
			rep_event_time += event_count[r];
			rep_gradm_event += event_count[r];
		}
	}
}
