use num_complex::Complex32;
use std::cell::RefCell;
use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

pub type RcDist = Rc<RefCell<Distribution>>;

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum DistType {
	P,
	Z,
	Z0,
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub enum DistRelation {
	PP,
	PZ,
	ZP,
	ZZ,
	MP,
	MZ,
}

pub static DIST_TYPE_STR: [&str; 3] = ["+", "z", "z0"];
pub static DIST_RELATION_STR: [&str; 6] = ["++", "+z", "z+", "zz", "-+", "-z"];

pub struct Edge {
	pub relation: DistRelation,
	pub rot_mat_factor: Complex32,
	pub dist: RcDist,
}

pub struct Distribution {
	pub mag: Complex32,
	pub signal: f32,
	pub rel_signal: f32, // relative to the strongest dist in the repetition
	pub kt_vec: [f32; 4],
	pub dist_type: DistType,
	pub ancestors: Vec<Edge>,
	pub influence: f32,     // signal change if this distribution is removed
	pub rel_influence: f32, // relative to max influence of all dists
}

impl Distribution {
	fn new(dist_type: DistType) -> Distribution {
		Distribution {
			mag: Complex32::new(0.0, 0.0),
			signal: 0.0,
			rel_signal: 0.0,
			kt_vec: [0.0, 0.0, 0.0, 0.0],
			dist_type,
			ancestors: Vec::new(),
			influence: 0.0,
			rel_influence: 0.0,
		}
	}

	fn measure(&mut self, data_shape: [f32; 3], mean_t2dash: f32, adc_phase: f32) {
		let sinc = |x: f32| if x == 0.0 { 1.0 } else { x.sin() / x };
		let pi = std::f32::consts::PI;
		let signal = self.mag
			* (-(self.kt_vec[3] / mean_t2dash).abs()).exp()
			* sinc(pi * self.kt_vec[0] / data_shape[0])
			* sinc(pi * self.kt_vec[1] / data_shape[1])
			* sinc(pi * self.kt_vec[2] / data_shape[2]);
		self.signal += (Complex32::new(0.0, adc_phase).exp() * signal).norm();
	}
}

pub struct Repetition {
	pub pulse_angle: f32,
	pub pulse_phase: f32,
	pub event_count: u32,
	pub event_time: Vec<f32>,
	pub gradm_event: Vec<[f32; 3]>,
	pub adc_phase: Vec<f32>,
	pub adc_mask: Vec<bool>,
}

pub fn compute_graph(
	seq: &Vec<Repetition>,
	data_shape: [f32; 3],
	mean_t1: f32,
	mean_t2: f32,
	mean_t2dash: f32,
	max_dist_count: usize,
	min_dist_mag: f32,
) -> Vec<Vec<RcDist>> {
	let mut graph: Vec<Vec<RcDist>> = Vec::new();

	let mut dists_p = Vec::<RcDist>::new();
	let mut dists_z = Vec::<RcDist>::new();
	let mut dist_z0 = Rc::new(RefCell::new(Distribution {
		mag: Complex32::new(1.0, 0.0),
		..Distribution::new(DistType::Z0)
	}));

	graph.push(vec![dist_z0.clone()]);

	for rep in seq.iter() {
		{
			let (_dists_p, _dists_z, _dist_z0) = apply_pulse(
				&dists_p,
				&dists_z,
				&dist_z0,
				&rep,
				max_dist_count,
				min_dist_mag,
			);
			dists_p = _dists_p;
			dists_z = _dists_z;
			dist_z0 = _dist_z0;
		}
		graph.push(
			iter::once(&dist_z0)
				.chain(&dists_p)
				.chain(&dists_z)
				.cloned()
				.collect(),
		);

		for e in 0..rep.event_count as usize {
			let dt = rep.event_time[e];
			let r1 = (-dt / mean_t1).exp();
			let r2 = (-dt / mean_t2).exp();

			for dist in dists_p.iter() {
				dist.borrow_mut().mag *= r2;
			}
			for dist in dists_z.iter() {
				dist.borrow_mut().mag *= r1;
			}
			let z0_mag = dist_z0.borrow().mag;
			dist_z0.borrow_mut().mag = z0_mag * r1 + (1.0 - r1);

			for dist in dists_p.iter() {
				dist.borrow_mut().kt_vec[0] += rep.gradm_event[e][0];
				dist.borrow_mut().kt_vec[1] += rep.gradm_event[e][1];
				dist.borrow_mut().kt_vec[2] += rep.gradm_event[e][2];
				dist.borrow_mut().kt_vec[3] += dt;

				if rep.adc_mask[e] {
					dist.borrow_mut()
						.measure(data_shape, mean_t2dash, rep.adc_phase[e]);
				}
			}
		}
	}
	graph
}

fn apply_pulse(
	dists_p: &Vec<RcDist>,
	dists_z: &Vec<RcDist>,
	dist_z0: &RcDist,
	rep: &Repetition,
	max_dist_count: usize,
	min_dist_mag: f32,
) -> (Vec<RcDist>, Vec<RcDist>, RcDist) {
	let angle = rep.pulse_angle;
	let phase = rep.pulse_phase;
	// Unaffected magnetisation
	let z_to_z = Complex32::from(angle.cos());
	let p_to_p = Complex32::from((angle / 2.0).cos().powi(2));
	// Excited magnetsiation
	let z_to_p = -0.70710678118 * Complex32::i() * angle.sin() * (Complex32::i() * phase).exp();
	let p_to_z = -z_to_p.conj();
	// Refocussed magnetisation
	let m_to_z = -z_to_p;
	let m_to_p = (1.0 - p_to_p) * (Complex32::i() * 2.0 * phase).exp();

	let mut dist_dict_p: HashMap<[i32; 4], RcDist> = HashMap::new();
	let mut dist_dict_z: HashMap<[i32; 4], RcDist> = HashMap::new();

	let mut add_dist = |kt_vec: [f32; 4],
						mag: Complex32,
						rot_mat_factor: Complex32,
						relation: DistRelation,
						ancestor: &RcDist| {
		let key = [
			(kt_vec[0] * 1e3).round() as i32,
			(kt_vec[1] * 1e3).round() as i32,
			(kt_vec[2] * 1e3).round() as i32,
			(kt_vec[3] * 1e6).round() as i32,
		];
		let dist_type = match relation {
			DistRelation::PP | DistRelation::MP | DistRelation::ZP => DistType::P,
			DistRelation::PZ | DistRelation::MZ | DistRelation::ZZ => DistType::Z,
		};
		let dist_dict = if dist_type == DistType::P {
			&mut dist_dict_p
		} else {
			&mut dist_dict_z
		};
		let mag = mag * rot_mat_factor;

		match dist_dict.get(&key) {
			Some(dist) => {
				dist.borrow_mut().mag += mag;
				dist.borrow_mut().ancestors.push(Edge {
					relation,
					rot_mat_factor,
					dist: ancestor.clone(),
				});
			}
			None => {
				if mag.norm() > min_dist_mag {
					let mut dist = Distribution {
						mag,
						kt_vec,
						..Distribution::new(dist_type)
					};
					dist.ancestors.push(Edge {
						relation,
						rot_mat_factor,
						dist: ancestor.clone(),
					});
					dist_dict.insert(key, Rc::new(RefCell::new(dist)));
				};
			}
		};
	};

	for dist in iter::once(dist_z0).chain(dists_z.iter()) {
		let mag = dist.borrow().mag;
		let kt_vec = dist.borrow().kt_vec;
		add_dist(kt_vec, mag, z_to_z, DistRelation::ZZ, dist);
		add_dist(kt_vec, mag, z_to_p, DistRelation::ZP, dist);
	}

	for dist in dists_p.iter() {
		let mag = dist.borrow().mag;
		let kt_vec = dist.borrow().kt_vec;
		add_dist(kt_vec, mag, p_to_p, DistRelation::PP, dist);
		add_dist(kt_vec, mag, p_to_z, DistRelation::PZ, dist);
		let mag = mag.conj();
		let kt_vec = [-kt_vec[0], -kt_vec[1], -kt_vec[2], -kt_vec[3]];
		add_dist(kt_vec, mag, m_to_p, DistRelation::MP, dist);
		add_dist(kt_vec, mag, m_to_z, DistRelation::MZ, dist);
	}

	let dist_z0 = match dist_dict_z.remove(&[0, 0, 0, 0]) {
		Some(dist) => {
			dist.borrow_mut().dist_type = DistType::Z0;
			dist
		}
		None => {
			let dist = Rc::new(RefCell::new(Distribution::new(DistType::Z0)));
			// Add a relation to the previous z0 state
			dist.borrow_mut().ancestors.push(Edge {
				relation: DistRelation::ZZ,
				rot_mat_factor: z_to_z,
				dist: dist_z0.clone(),
			});
			dist
		}
	};
	let mut dists_p: Vec<RcDist> = dist_dict_p.values().cloned().collect();
	let mut dists_z: Vec<RcDist> = dist_dict_z.values().cloned().collect();
	let mag = |dist: &RcDist| dist.borrow().mag.norm();
	dists_p.sort_unstable_by(|a, b| mag(&b).partial_cmp(&mag(&a)).unwrap()); // reversed sort
	dists_z.sort_unstable_by(|a, b| mag(&b).partial_cmp(&mag(&a)).unwrap()); // reversed sort
	dists_p.truncate(max_dist_count);
	dists_z.truncate(max_dist_count);

	(dists_p, dists_z, dist_z0)
}

pub fn simplify_graph(graph: &mut Vec<Vec<RcDist>>) {
	// Tell each dist how large its signal is relative to the strongest dist
	for rep in graph.iter() {
		let mut max_rep_signal: f32 = 0.0;

		for dist in rep.iter() {
			max_rep_signal = max_rep_signal.max(dist.borrow().signal);
		}

		for dist in rep.iter() {
			let signal = dist.borrow().signal;
			dist.borrow_mut().rel_signal = signal / max_rep_signal;
		}
	}
	// Calculate for every dist how much the total signal changes if it is removed
	// This code uses abs(complex mag) to estimate ratios, which can overestimate influences
	for rep in graph.iter().rev() {
		for dist in rep.iter() {
			// This code assumes that influence was 0 before these for-loops
			let signal = dist.borrow().signal;
			dist.borrow_mut().influence += signal;

			// To get the contribution of ancestors we need to compare every single
			// ancestor to the sum of all ancestors
			let mut total_mag = 0.0;
			for ancestor in dist.borrow().ancestors.iter() {
				total_mag += (ancestor.rot_mat_factor * ancestor.dist.borrow().mag).norm();
			}

			// This code overestimates the influences of ancestors of z0 distributions,
			// they regrow a part of their magnetisation themselves. (would need full separation)
			for ancestor in dist.borrow().ancestors.iter() {
				let contribution = (ancestor.rot_mat_factor * ancestor.dist.borrow().mag).norm();
				ancestor.dist.borrow_mut().influence +=
					dist.borrow().influence * contribution / total_mag;
			}
		}
	}
	// Search for the maximum influence of all dists to compute rel_influence
	let mut max_influence: f32 = 0.0;
	for rep in graph.iter() {
		for dist in rep.iter() {
			max_influence = max_influence.max(dist.borrow().influence);
		}
	}
	for rep in graph.iter() {
		for dist in rep.iter() {
			let influence = dist.borrow().influence;
			dist.borrow_mut().rel_influence = influence / max_influence;
		}
	}
}
