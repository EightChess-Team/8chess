mod inputs;

use inputs::{PieceFeatures8Chess, ThreatFeatures8Chess, piece_indices, threat_indices, stm_is_white, output_bucket};

use bullet_lib::{
    game::inputs::SparseInputType,
    trainer::schedule::lr,
    trainer::schedule::lr::LrScheduler,
    value::loader::DirectSequentialDataLoader,
};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelWeights, SavedFormat},
    optimiser::{Optimiser, adam::{AdamW, AdamWParams}},
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
    reader::ReadMapLoader,
    model::ModelInputsMapper,
};

use bulletformat::ChessBoard;

const L1: usize = 1024;
const L2: usize = 16;
const L3: usize = 32;
const OUTPUT_BUCKETS: usize = 8;
const PIECE_FEATURES: usize = 7680;
const THREAT_FEATURES: usize = 66864;
const Q_FT: i16 = 255;
const Q_L1: i16 = 64;

fn main() {
    eightchess::init_threat_indexing();

    let piece = PieceFeatures8Chess;
    let threat = ThreatFeatures8Chess;

    let inputs = ModelInputs::default()
        .add_sparse("stm/piece", (piece.num_inputs(), 1), piece.max_active())
        .add_sparse("ntm/piece", (piece.num_inputs(), 1), piece.max_active())
        .add_sparse("stm/threat", (threat.num_inputs(), 1), threat.max_active())
        .add_sparse("ntm/threat", (threat.num_inputs(), 1), threat.max_active())
        .add_sparse("buckets", (OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((((stm_piece, ntm_piece), stm_threat), ntm_threat), buckets), target)| {
            let l0p = builder.new_affine("l0p", PIECE_FEATURES, L1);
            let l0t = builder.new_affine("l0t", THREAT_FEATURES, L1);

            let l1 = builder.new_affine("l1", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, OUTPUT_BUCKETS);

            let ft = |piece, threat, start, end| {
                (l0p.slice(start, end).forward(piece) + l0t.slice(start, end).forward(threat)).crelu()
            };
            let stm_hidden = ft(stm_piece, stm_threat, 0, L1 / 2) * ft(stm_piece, stm_threat, L1 / 2, L1);
            let ntm_hidden = ft(ntm_piece, ntm_threat, 0, L1 / 2) * ft(ntm_piece, ntm_threat, L1 / 2, L1);
            let hl0 = stm_hidden.concat(ntm_hidden);

            let hl1 = l1.forward(hl0).select(buckets).screlu();
            let hl2 = l2.forward(hl1).select(buckets).screlu();
            let out = l3.forward(hl2).select(buckets);

            let loss = out.sigmoid().squared_error(target);
            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), out)])
        },
    );

    let weights = ModelWeights::new(&defn, 12345);
    let device = DefaultDevice::new(0).unwrap();
    let params = AdamWParams::default();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), params).unwrap();

    let saved_format = vec![
        SavedFormat::id("l0tw").transpose().transform(|_, v| v.iter().map(|x| x.clamp(-0.498, 0.498)).collect()).round().quantise::<i8>(Q_FT),
        SavedFormat::id("l0pw").transpose().transform(|_, v| v.iter().map(|x| x.clamp(-128.0, 128.0)).collect()).round().quantise::<i16>(Q_FT),
        SavedFormat::id("l0tb").transform(|store, values| {
            let p = store.get("l0pb").values.f32();
            values.iter().zip(p).map(|(a, b)| (a + b).clamp(-128.0, 128.0)).collect()
        }).round().quantise::<i16>(Q_FT),
        SavedFormat::id("l1w").transform(|_store, values| {
            let mut out = vec![0f32; values.len()];
            for b in 0..8usize {
                for o in 0..16usize {
                    for i in 0..1024usize {
                        let bullet_idx = i * 128 + (b * 16 + o);
                        let engine_idx = b * 16384 + (i / 4) * 64 + o * 4 + (i % 4);
                        out[engine_idx] = values[bullet_idx].clamp(-1.984375, 1.984375);
                    }
                }
            }
            out
        }).round().quantise::<i8>(Q_L1),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w").transpose(),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w").transpose(),
        SavedFormat::id("l3b"),
    ];

    let mapper = ModelInputsMapper::build(&inputs, |pos: &ChessBoard, _step, (((((stm_piece, ntm_piece), stm_threat), ntm_threat), buckets), target)| {
        let (pw, pb) = piece_indices(pos);
        let (tw, tb) = threat_indices(pos);
        let stm_w = stm_is_white(pos);

        let (stm_p, ntm_p) = if stm_w { (pw, pb) } else { (pb, pw) };
        let (stm_t, ntm_t) = if stm_w { (tw, tb) } else { (tb, tw) };

        for v in stm_piece.iter_mut() { *v = -1; }
        for (i, &idx) in stm_p.iter().enumerate() {
            if i < stm_piece.len() { stm_piece[i] = idx as i32; }
        }
        for v in ntm_piece.iter_mut() { *v = -1; }
        for (i, &idx) in ntm_p.iter().enumerate() {
            if i < ntm_piece.len() { ntm_piece[i] = idx as i32; }
        }
        for v in stm_threat.iter_mut() { *v = -1; }
        for (i, &idx) in stm_t.iter().enumerate() {
            if i < stm_threat.len() { stm_threat[i] = idx as i32; }
        }
        for v in ntm_threat.iter_mut() { *v = -1; }
        for (i, &idx) in ntm_t.iter().enumerate() {
            if i < ntm_threat.len() { ntm_threat[i] = idx as i32; }
        }
        buckets[0] = output_bucket(pos) as i32;
        // WDL target with lambda=0.5 (blend result and eval score)
        let result = pos.result as f32 / 2.0;
        let score = 1.0 / (1.0 + (-pos.score as f32 / 400.0).exp());
        target[0] = 0.5 * result + 0.5 * score;
    });

    let data_path = std::env::args().nth(1).unwrap_or_else(|| "data.bullet".to_string());
    let dataloader = DirectSequentialDataLoader::new(&[data_path.as_str()]);
    let readmap = ReadMapLoader::new(dataloader, mapper, 32);

    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size: 16384,
            batches_per_superbatch: 100,
            start_superbatch: 1,
            end_superbatch: 800,
        },
        lr_schedule: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.00003, final_superbatch: 800 }.boxed(),
        log_rate: 64,
        track_grad_norms: false,
    };

    train(
        &mut optimiser,
        schedule,
        readmap,
        |_, _, _| {},
        |optimiser, step| {
            println!("superbatch {}", step.superbatch());
            if step.superbatch() % 10 == 0 || step.superbatch() == step.final_superbatch() {
                std::fs::create_dir_all("checkpoints").unwrap_or(());
                let name = format!("checkpoints/sb{}", step.superbatch());
                bullet_lib::value::save::save_to_checkpoint(optimiser, &saved_format, &name);
                println!("saved {}", name);
            }
        },
    )
    .unwrap();

    // final export: trim quantised.bin to engine size
    std::fs::create_dir_all("checkpoints/final").unwrap_or(());
    bullet_lib::value::save::save_to_checkpoint(&optimiser, &saved_format, "checkpoints/final");
    let q = std::fs::read("checkpoints/final/quantised.bin").unwrap();
    std::fs::write("eightchess_bullet.nnue", &q[..84349504]).unwrap();
    println!("exported eightchess_bullet.nnue");
}
