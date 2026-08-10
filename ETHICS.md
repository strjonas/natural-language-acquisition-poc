# Ethics and responsible claims

## Release assessment

Publishing this repository is currently low risk. It contains a small simulated
gridworld, standard learning methods, compact experiment artifacts, and no
personal, medical, biometric, or human-participant data. It does not provide a
meaningful capability for physical harm or autonomous deployment.

The main ethical risk is epistemic: anthropomorphizing a narrow estimator or
presenting a selected three-word report as evidence of consciousness, sentience,
or human-like language. This repository therefore uses functional terms such as
*body-state estimator*, *individual-body model*, and *report policy*. “Self” is
a research hypothesis tested through operational criteria, not an ontological
claim.

## Claim boundaries

The present evidence supports a limited claim: in this simulator, a standard
online estimator can identify hidden constants of one individual body from
intermittent readings, maintain a better body-state belief between readings,
and improve a constrained report. It does not establish:

- consciousness, experience, personhood, or moral patienthood;
- natural-language development or open-ended symbol emergence;
- discovery of the body-model architecture from raw experience;
- robustness to real sensors, actuators, distribution shift, or people;
- safe autonomous operation on a physical robot.

## Research and publication practices

- AI assistance is disclosed in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
- Strong experiments have dated preregistrations and append-only result records.
- Failed gates and negative experiments remain in the public history.
- Saved public artifacts are allow-listed; local caches and the large run archive
  remain excluded.
- Novelty statements should say “no exact match found in a targeted search,”
  never “the first,” unless a systematic review supports that stronger claim.

## Future work

Moving from simulation to physical robots requires independent safeguards:
bounded actuation, emergency stopping, collision avoidance outside the learned
policy, logging, staged testing, and human supervision. Work involving people,
personal dialogue, or collected interaction traces may require informed consent,
privacy controls, and institutional ethics review. Systems that elicit social
attachment should avoid deceptive claims about inner experience.
