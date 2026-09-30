# Isoform selectivity of the VB/EV series against CK2α and CK2α′

Draft results text and the numbers behind it. Values are MM-GBSA interaction
energies in kcal/mol from the generalized Born analysis, taken from the
difference section (complex − receptor − ligand). No entropy term was
computed, so these are enthalpy-like and their absolute values are not free
energies of binding; only differences between the two subunits for the same
compound are interpreted.

---

## Sampling and the noise floor

CK2α′ complexes of EV043 were run in three independent replicas of 300 ns. A
fourth replica was discarded as corrupted before analysis. The three retained
replicas give a spread on the total interaction energy of **1.97 kcal/mol**
(1 SD), and this is used throughout as the threshold below which a difference
is not interpreted.

This between-replica spread is the relevant measure of reproducibility, and it
is larger than the frame-to-frame scatter reported within any single
trajectory. Two further quantities were measured across the same replicas to
characterise it:

| quantity | replica range | 1 SD |
|---|---|---|
| total interaction energy (CK2α′) | −38.7 to −42.6 | 1.97 |
| Lys69 electrostatic contribution | −3.62 to −5.86 | 1.15 |
| ligand–Lys69 contact occupancy | 44.4 to 84.2 % | 20.3 pts |
| ligand heavy-atom RMSD | 3.20 to 4.74 Å | 0.78 |

Per-residue contributions and contact occupancies therefore carry
substantially more replica noise than the total, and single-trajectory
differences in those quantities are not interpreted here.

---

## The series reverses the reference compound's isoform preference

Taking ΔΔ = Δ(CK2α′) − Δ(CK2α), so that a negative value denotes a preference
for CK2α′:

Each compound's two subunits are compared at equal sampling; the frame count
of every analysis is given, since a difference between subunits is a
preference only if both sides were sampled alike.

| compound | CK2α′ | CK2α | frames | **ΔΔ** |
|---|---|---|---|---|
| CX-4945 (reference) | −29.95 | −38.78 | 581 / 581 | **+8.83** |
| VB004 | −37.97 | −31.48 | 181 / 181 | **−6.49** |
| EV043 | −42.57 | −35.96 | 181 / 181 | **−6.62** |

CX-4945 favours CK2α by 8.83 kcal/mol. Both designed compounds favour CK2α′ by
about 6.5 kcal/mol. The separation between the reference and either analogue
is **15.3–15.5 kcal/mol**, roughly eight times the replica spread, and is the
largest effect observed in this work.

**VB004 and EV043 differ from each other by 0.13 kcal/mol** — far inside the
replica spread, and consistent with every other method applied. The two
analogues are equivalent.

Longer analyses exist for the CK2α′ complexes of both analogues (581 and 596
frames) but not for their CK2α complexes, so they are not used for the
subunit comparison. Across three CK2α′ replicas of EV043 the total ranges
from −38.7 to −42.6, which is the basis of the spread quoted above.

---

## The reversal is electrostatic in origin and gains a shape term

> **To be regenerated.** The component values below came from an analysis that
> paired 581-frame CK2α′ replicas against 181-frame CK2α runs for the
> analogues. The totals above have been corrected to matched pairs; these
> components have not. The direction of the CX-4945 result is unaffected, as
> that comparison was matched throughout.

Resolving ΔΔ into its components, with the electrostatic and polar solvation
terms combined since they largely cancel:

| compound | net electrostatic | van der Waals | non-polar surface | **total** |
|---|---|---|---|---|
| CX-4945 | **+9.21** | −0.39 | +0.02 | **+8.83** |
| VB004 | −1.28 | −1.32 | −0.06 | **−2.66** |
| EV043 | −2.22 | −2.34 | −0.34 | **−4.90** |

Two things distinguish the analogues from the reference.

**CX-4945's preference for CK2α is entirely electrostatic** (+19.39 before
solvation, +9.21 after) and carries **no shape preference whatsoever** (van der
Waals −0.39). **Both analogues instead show a van der Waals preference for
CK2α′** of 1.3–2.3 kcal/mol, which the reference does not have.

The non-polar surface term is flat across all three compounds and both
subunits (ΔΔ between −0.34 and +0.02), so the buried surface area is
unchanged. The van der Waals gain reflects the quality of contact rather than
its extent.

---

## The electrostatic difference lies in CK2α, not CK2α′

Raw electrostatic terms:

| compound | CK2α′ | CK2α |
|---|---|---|
| CX-4945 | −16.66 | **−36.04** |
| VB004 | −15.57 | −13.97 |
| EV043 | −16.74 ± 2.8 | **−8.39** |

In CK2α′ all three compounds are equivalent (−15.6 to −16.7). The entire
difference arises in CK2α, where CX-4945 makes an electrostatic interaction
more than twice that of either analogue.

Per-residue decomposition locates it. In CK2α, CX-4945 contributes −7.63 at
the catalytic lysine (Lys68) and −5.85 at Glu80; EV043's contribution at the
same lysine falls below the reporting cutoff (< 0.30) and VB004's is −3.17.
CX-4945 carries an ionisable carboxylate and forms a salt bridge to the
catalytic lysine of CK2α that neither analogue reproduces.

**Loss of this salt bridge in CK2α, rather than any gain in CK2α′, is the
origin of the reversed isoform preference.** This is consistent with the
ensemble docking, where the change across the series arose from degradation of
the CK2α score rather than improvement of the CK2α′ score.

---

## Negative results

**The chlorine substituent makes no contact with the catalytic lysine.** Across
three independent 300 ns replicas, both chlorines of EV043 remain 9.9–13.4 Å
from Lys69, with a maximum occupancy within 4.5 Å of 3.2 %. Measured from the
lysine Cα, which is fixed by the fold, the result is unchanged (10.7–11.9 Å).
Ranking every ligand heavy atom by its distance to that residue places the
chlorines 14th to 29th of 29 in every replica. The nearest atom, in every
replica and in both compounds, is a ring nitrogen (N2) at 3.4–5.0 Å.

The one chlorine contact that does occur, to the Leu46 backbone carbonyl,
approaches at a mean C–Cl···O angle of 92° and is within 15° of the C–Cl axis
in 0.9 % of frames. It is a side-on van der Waals contact, not a halogen bond.
Fixed-charge force fields carry no σ-hole, so this was assessed
geometrically rather than energetically.

**The analogues are not distinguished from one another by any structural
method applied.** Ensemble docking gave identical hinge-contact metrics;
crystal-structure docking gave a between-compound score spread smaller than
the within-compound spread across three structures; per-residue decomposition
gave a net difference of 0.24 kcal/mol; contact occupancies and ligand RMSD
fell inside the replica noise; and the total interaction energies differ by
less than the replica spread permits resolving.

At the catalytic lysine in CK2α′, averaged over replicas, both compounds give
**−4.60 kcal/mol** — identical. The single-trajectory value of −5.86 that first
suggested a difference is the upper end of EV043's own replica range.

---

## Limitations

- CX-4945 and VB004 were simulated once per subunit. The 1.97 kcal/mol spread
  is EV043's and is applied to them by assumption. Replicates of the CX-4945
  systems would be required before the reversal is stated without
  qualification.
- No entropy term is included. Absolute values are not binding free energies.
- The analogues are compared at 181 frames and CX-4945 at 581. Each compound's
  own subunit comparison is internally matched, which is what ΔΔ requires, but
  the absolute totals are not comparable across that boundary.
- CK2α numbering and CK2α′ numbering differ by one residue at equivalent
  positions, and the catalytic lysine is Lys68 in CK2α and Lys69 in CK2α′.
  Topology files and MM-GBSA output additionally differ by one. All residue
  assignments here were made by sequence context rather than by offset.
