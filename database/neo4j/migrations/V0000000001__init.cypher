// (p1:Person)-[KNOWS]->(p2:Person)
// - Person
//   - name: str, null
//   - user_id: int, null
// - KNOWS
//   - type: associate_of, child_of, parent_of, null
//   - reputation: positive, negative, neutral

// (p1:Person)-[WITH]->(p2:Person)
// - WITH
//   - verb: *

// (p1:Person)-[AT]->(p2:Place)

// (p1:Person)-[CAN_DO]->(sk1:Skill)

// (p1:Person)-[MEMBER_OF]->(pg1:PeopleGroup)
// - PeopleGroup
//   - name: str, null
//   - hierarchy: centralized, decentralized, null
//   - scope: ethnic, local, global, national, provincial
