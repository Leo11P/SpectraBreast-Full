# fairino_description

Qui vanno l'URDF e le mesh del FR5, presi da `frcobot_ros2` (cartella
`fairino_description` del pacchetto ufficiale Fairino):

```
fairino_description/
├── urdf/
│   └── fairino5_v6.urdf
└── meshes/
    └── fairino5_v6/
        ├── base_link.STL
        ├── shoulder_link.STL
        ├── upperarm_link.STL
        ├── forearm_link.STL
        ├── wrist1_link.STL
        ├── wrist2_link.STL
        └── wrist3_link.STL
```

Il collision checker (`src/collision_fr5.py`) risolve i path
`package://fairino_description/...` dell'URDF rispetto a questa cartella.
Se la sposti, aggiorna `paths.urdf` in `config.yaml`.
