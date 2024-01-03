# Vulkano 101
## 1. Créer une instance Vulkan
Vulkan n'est pas une implémentation universelle, chaque driver de carte graphique implémente les fonctions comme il veut, on a juste des headers C++ en commun. Il faut donc charger depuis Rust les binaires d'implémentation (.dll/.so).

```rust
let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
```
On peut alors construire une instance Vulkan, c'est à dire une instance de communication avec le driver.

```rust
let instance = Instance::new(library, InstanceCreateInfo::default())
    .expect("failed to create instance");
```

## 2. Création de devices
### Physical device
Avant de commencer, il faut choisir avec quel périphérique on souhaite travailler (dans le cas où la machine ait plusieurs GPU compatibles avec Vulkan).

```rust
let physical_device = instance
    .enumerate_physical_devices()
    .expect("could not enumerate devices")
    .next()
    .expect("no devices available");
```

Ici, on choisit le premier dans la liste pour des raisons de simplicité.
(cf. Enumerating physical devices pour plus d'informations)

### Logical device
Après avoir sélectionné un périphérique, il faut initialiser un tunnel de communication avec le périphérique sous la forme d'un `Device`.

Pour créer un device, il faut d'abord lister les différentes queues que le périphérique a. Une *queue* est l'équivalent pour un GPU d'un thread materiel pour le CPU.

Chaque *queue* a des capacités différentes, certaines peuvent faire du calcul graphique alors que d'autres ne peuvent être utilisées que pour des allocations de mémoire.

```rust
let queue_family_index = physical_device
    .queue_family_properties()
    .iter()
    .enumerate()
    .position(|(_queue_family_index, queue_family_properties)| {
        queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
    })
    .expect("couldn't find a graphical queue family") as u32;
```

Après avoir obtenu la liste des *queues* capables de faire du calcul graphique, il ne reste plus qu'a créer le tunnel de communication.

```rust
let (device, mut queues) = Device::new(
    physical_device,
    DeviceCreateInfo {
        // here we pass the desired queue family to use by index
        queue_create_infos: vec![QueueCreateInfo {
            queue_family_index,
            ..Default::default()
        }],
        ..Default::default()
    },
)
.expect("failed to create device");

let queue = queues.next().unwrap();
```

