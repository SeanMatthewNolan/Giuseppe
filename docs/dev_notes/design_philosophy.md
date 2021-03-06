# Design Philosophy

## General Discussion

This project will generally use object-oriented programming (OOP).
Though in recent years, many discourage OOP, OOP is best suited to the "structural" nature of the necessary problem transformations.

## Guidelines

Nothing is strictly binding, but for consistency and to avoid beluga's issues, these showed serve as guidelines when developing Giuseppe:

1. Keep it simple first; add more features later (focus on the problem at hand)
2. Retain problem information as long as possible
3. Have classes "ingest" information during initialization (exclude "input" classes) to avoid objects in "incomplete" state
4. The "seed guess" should be solved prior to starting a continuation series
