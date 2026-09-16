//! A typed in-process publish and subscribe bus.
//!
//! Frames route by their Rust type, which is what the Python version did
//! with `type(frame)` and a dictionary. Each subscriber holds its own
//! bounded queue, and a publisher never blocks: when a queue is full the
//! frame is dropped for that subscriber alone.
//!
//! Dropping rather than blocking is deliberate. A slow consumer, which in
//! practice means a renderer, must not be able to stall a control loop,
//! and a bounded queue is what makes the memory a pipeline uses
//! predictable. See the bounded-resource rules in the defensive
//! guideline.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

/// Default queue capacity per subscriber.
pub const DEFAULT_CAPACITY: usize = 64;

/// What one publish did.
///
/// The Python bus dropped silently, so a pipeline losing frames looked
/// exactly like a pipeline producing none. Reporting the count costs
/// nothing and makes the loss measurable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PublishReport {
    /// Subscribers that accepted the frame.
    pub delivered: usize,
    /// Subscribers whose queue was full, so the frame was dropped.
    pub dropped: usize,
}

/// The queue behind one subscription.
#[derive(Debug)]
struct Queue<T> {
    frames: Mutex<VecDeque<T>>,
    capacity: usize,
}

/// A subscriber's end of the bus, for one frame type.
///
/// Dropping this unsubscribes: the bus holds a weak reference, so a
/// consumer that goes away stops costing a publisher anything.
#[derive(Debug)]
pub struct Subscription<T> {
    queue: Arc<Queue<T>>,
}

impl<T> Subscription<T> {
    /// Takes the oldest frame, if any.
    #[must_use]
    pub fn try_next(&self) -> Option<T> {
        self.queue
            .frames
            .lock()
            .ok()
            .and_then(|mut frames| frames.pop_front())
    }

    /// Takes the newest frame, discarding anything older.
    ///
    /// What a renderer wants: the current state rather than the backlog.
    #[must_use]
    pub fn drain_latest(&self) -> Option<T> {
        let mut frames = self.queue.frames.lock().ok()?;
        let latest = frames.pop_back();
        frames.clear();
        latest
    }

    /// How many frames are waiting.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.frames.lock().map_or(0, |frames| frames.len())
    }

    /// Whether nothing is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An in-process bus routing frames by type.
///
/// # Examples
///
/// ```
/// use arco_runtime::bus::Bus;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct Pose(f64, f64);
///
/// let bus = Bus::new(8);
/// let poses = bus.subscribe::<Pose>();
/// let report = bus.publish(&Pose(1.0, 2.0));
///
/// assert_eq!(report.delivered, 1);
/// assert_eq!(poses.try_next(), Some(Pose(1.0, 2.0)));
/// ```
#[derive(Debug, Default)]
pub struct Bus {
    subscribers: Mutex<BTreeMap<TypeId, Vec<Weak<dyn Any + Send + Sync>>>>,
    capacity: usize,
}

impl Bus {
    /// Builds a bus whose subscriber queues hold `capacity` frames.
    ///
    /// A capacity of zero means unbounded, matching the Python default of
    /// `queue.Queue(maxsize=0)`. Use it knowing that a stalled consumer
    /// then grows without limit.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            subscribers: Mutex::new(BTreeMap::new()),
            capacity,
        }
    }

    /// Builds a bus with the default capacity.
    #[must_use]
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }

    /// The queue capacity every subscription gets.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Subscribes to frames of type `T`.
    ///
    /// Safe at any time, including after a pipeline has started, which is
    /// the late-subscriber support the Python bus advertised.
    pub fn subscribe<T: Any + Send + Sync>(&self) -> Subscription<T> {
        let queue = Arc::new(Queue::<T> {
            frames: Mutex::new(VecDeque::new()),
            capacity: self.capacity,
        });

        if let Ok(mut subscribers) = self.subscribers.lock() {
            let erased: Arc<dyn Any + Send + Sync> = queue.clone();
            subscribers
                .entry(TypeId::of::<T>())
                .or_default()
                .push(Arc::downgrade(&erased));
        }

        Subscription { queue }
    }

    /// How many live subscribers there are for type `T`.
    #[must_use]
    pub fn subscriber_count<T: Any + Send + Sync>(&self) -> usize {
        self.subscribers.lock().map_or(0, |subscribers| {
            subscribers.get(&TypeId::of::<T>()).map_or(0, |queues| {
                queues.iter().filter(|q| q.strong_count() > 0).count()
            })
        })
    }

    /// Broadcasts a frame to every subscriber of its type.
    ///
    /// Never blocks. A subscriber whose queue is full does not receive
    /// this frame, and the returned report says how many did not.
    pub fn publish<T: Any + Clone + Send + Sync>(&self, frame: &T) -> PublishReport {
        let Ok(mut subscribers) = self.subscribers.lock() else {
            return PublishReport::default();
        };
        let Some(queues) = subscribers.get_mut(&TypeId::of::<T>()) else {
            return PublishReport::default();
        };

        // A subscription that has been dropped leaves a dead weak
        // reference behind, and clearing them here is what keeps a
        // long-running pipeline from accumulating them.
        queues.retain(|queue| queue.strong_count() > 0);

        let mut report = PublishReport::default();
        for weak in queues.iter() {
            let Some(erased) = weak.upgrade() else {
                continue;
            };
            let Ok(queue) = Arc::downcast::<Queue<T>>(erased) else {
                continue;
            };
            let Ok(mut frames) = queue.frames.lock() else {
                continue;
            };
            if queue.capacity > 0 && frames.len() >= queue.capacity {
                report.dropped = report.dropped.saturating_add(1);
            } else {
                frames.push_back(frame.clone());
                report.delivered = report.delivered.saturating_add(1);
            }
        }
        report
    }
}
