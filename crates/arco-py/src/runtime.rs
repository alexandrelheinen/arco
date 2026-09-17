//! `arco.middleware` and `arco.pipeline` as Python still sees them.
//!
//! Two import paths, one crate behind them, which is deviation A-05. The
//! names, the argument order and the defaults are the ones `docs/API.md`
//! records, per `FR-API-01` and `FR-API-02`.
//!
//! Three of these classes are bases a caller subclasses rather than
//! objects a caller only calls. `Bus`, `BusPublisher` and `BusSubscriber`
//! are declared `subclass`, and `PipelineNode` both extends
//! `BusPublisher` and is itself subclassed, because a node is written by
//! overriding `run`. Deviation A-24 records that a Python override is
//! honored by calling back into the interpreter, the same cost the
//! keyword hooks carry under A-07.
//!
//! What the runtime crate supplies here is the node lifecycle:
//! [`arco_runtime::node::Handle`] owns the thread, catches a panic that
//! would otherwise cross the boundary, and records how the run ended,
//! which the Python runner could not do because it swallowed every
//! exception its nodes raised.
//!
//! Routing stays in this module rather than in [`arco_runtime::bus::Bus`],
//! and the reason is a type system rather than a preference. That bus
//! routes on `TypeId`, a Rust type known when the code is compiled, while
//! a frame type here is a Python class chosen at run time. The Python
//! contract also says `subscribe` hands back a `queue.Queue`, which a
//! caller then polls with `get_nowait` and guards with `queue.Empty`, and
//! a `Subscription` is not one. So the registry below holds real queues
//! and routes on the identity of the class object, which is what
//! `type(frame)` compared in the first place. The drop accounting is the
//! crate's [`PublishReport`] all the same, so a pipeline losing frames is
//! measurable rather than silent.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arco_core::Error;
use arco_runtime::bus::{DEFAULT_CAPACITY, PublishReport};
use arco_runtime::node::{Control, Handle, Node, Outcome};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::errors::OrRaise;

/// Reports a lock whose holder panicked, rather than panicking again.
fn unusable(what: &'static str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("the {what} is unusable"))
}

// ---------------------------------------------------------------------
// The bus
// ---------------------------------------------------------------------

/// Abstract base for the shared in-memory message bus.
///
/// Subclasses implement `publish`, `subscribe` and `subscriber_count`.
/// Python declared these abstract through `abc`; here each one raises
/// `NotImplementedError` when it is reached, so a subclass that forgot
/// one fails loudly at the call rather than returning a quiet nothing.
#[pyclass(subclass, name = "Bus", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyBus;

#[pymethods]
impl PyBus {
    /// Absorbs a subclass calling `super().__init__(...)`.
    ///
    /// A compiled class does its construction in `__new__`, so `__init__`
    /// falls through to `object.__init__`, which refuses arguments. A
    /// Python subclass forwarding its own arguments upward then fails on
    /// a line that worked against the pure-Python base.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Builds the base, which carries no state of its own.
    #[new]
    #[pyo3(text_signature = "()")]
    const fn new() -> Self {
        Self
    }

    /// Publishes `frame` to every subscriber registered for its type.
    #[pyo3(text_signature = "(frame)")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn publish(&self, _frame: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Bus.publish is abstract.",
        ))
    }

    /// Registers a new subscriber queue for `frame_type`.
    #[pyo3(text_signature = "(frame_type)")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn subscribe(&self, _frame_type: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Bus.subscribe is abstract.",
        ))
    }

    /// The number of active subscribers for `frame_type`.
    #[pyo3(text_signature = "(frame_type)")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn subscriber_count(&self, _frame_type: &Bound<'_, PyAny>) -> PyResult<usize> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Bus.subscriber_count is abstract.",
        ))
    }
}

/// Every queue registered for one frame type.
#[derive(Debug)]
struct Subscribers {
    frame_type: Py<PyAny>,
    queues: Vec<Py<PyAny>>,
}

/// Thread-safe in-process bus backed by bounded queues.
///
/// Frames route by the identity of their class, which is what the Python
/// implementation compared when it looked `type(frame)` up in a
/// dictionary. A subscriber whose queue is full does not receive the
/// frame, so a slow consumer never blocks a producer.
#[pyclass(extends = PyBus, subclass, name = "InMemoryBus", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyInMemoryBus {
    registry: Mutex<Vec<Subscribers>>,
    report: Mutex<PublishReport>,
    /// The capacity every subscriber queue is created with.
    #[pyo3(get)]
    maxsize: usize,
}

#[pymethods]
impl PyInMemoryBus {
    /// Builds a bus whose subscriber queues hold `maxsize` frames.
    ///
    /// A capacity of zero means unbounded, which is what
    /// `queue.Queue(maxsize=0)` means and what a stalled consumer then
    /// grows without limit under.
    #[new]
    #[pyo3(signature = (maxsize = DEFAULT_CAPACITY))]
    #[pyo3(text_signature = "(maxsize=64)")]
    fn new(maxsize: usize) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyBus).add_subclass(Self::over(maxsize))
    }

    /// Publishes `frame` to every subscriber registered for its type.
    ///
    /// Never blocks. A full queue drops the frame for that subscriber
    /// alone, and `last_publish` says how many were dropped.
    #[pyo3(text_signature = "(frame)")]
    fn publish(&self, py: Python<'_>, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        let frame_type = frame.get_type();
        let queues = {
            let registry = self.registry.lock().map_err(|_poisoned| unusable("bus"))?;
            registry
                .iter()
                .find(|entry| entry.frame_type.bind(py).is(&frame_type))
                .map(|entry| {
                    entry
                        .queues
                        .iter()
                        .map(|queue| queue.clone_ref(py))
                        .collect::<Vec<Py<PyAny>>>()
                })
                .unwrap_or_default()
        };

        let full = py.import("queue")?.getattr("Full")?;
        let mut report = PublishReport::default();
        for queue in &queues {
            match queue.bind(py).call_method1("put_nowait", (frame,)) {
                Ok(_delivered) => report.delivered = report.delivered.saturating_add(1),
                Err(raised) if raised.is_instance(py, &full) => {
                    // A slow consumer, which in practice means a renderer.
                    // Dropping is deliberate: a bounded queue is what keeps
                    // the memory a pipeline uses predictable.
                    report.dropped = report.dropped.saturating_add(1);
                }
                Err(raised) => return Err(raised),
            }
        }
        if let Ok(mut last) = self.report.lock() {
            *last = report;
        }
        Ok(())
    }

    /// Registers and returns a new subscriber queue for `frame_type`.
    ///
    /// Safe to call at any time, including after a pipeline has started,
    /// which is the late-subscriber support the Python bus advertised.
    #[pyo3(text_signature = "(frame_type)")]
    fn subscribe(&self, py: Python<'_>, frame_type: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let arguments = PyDict::new(py);
        arguments.set_item("maxsize", self.maxsize)?;
        let queue = py
            .import("queue")?
            .getattr("Queue")?
            .call((), Some(&arguments))?;

        let mut registry = self.registry.lock().map_err(|_poisoned| unusable("bus"))?;
        match registry
            .iter_mut()
            .find(|entry| entry.frame_type.bind(py).is(frame_type))
        {
            Some(entry) => entry.queues.push(queue.clone().unbind()),
            None => registry.push(Subscribers {
                frame_type: frame_type.clone().unbind(),
                queues: vec![queue.clone().unbind()],
            }),
        }
        Ok(queue.unbind())
    }

    /// The number of subscriber queues registered for `frame_type`.
    #[pyo3(text_signature = "(frame_type)")]
    fn subscriber_count(&self, py: Python<'_>, frame_type: &Bound<'_, PyAny>) -> PyResult<usize> {
        let registry = self.registry.lock().map_err(|_poisoned| unusable("bus"))?;
        Ok(registry
            .iter()
            .find(|entry| entry.frame_type.bind(py).is(frame_type))
            .map_or(0, |entry| entry.queues.len()))
    }

    /// Removes a previously registered subscriber queue.
    ///
    /// A queue that is not registered for `frame_type` is left alone.
    #[pyo3(text_signature = "(frame_type, q)")]
    fn unsubscribe(
        &self,
        py: Python<'_>,
        frame_type: &Bound<'_, PyAny>,
        q: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut registry = self.registry.lock().map_err(|_poisoned| unusable("bus"))?;
        if let Some(entry) = registry
            .iter_mut()
            .find(|entry| entry.frame_type.bind(py).is(frame_type))
        {
            entry.queues.retain(|queue| !queue.bind(py).is(q));
        }
        Ok(())
    }

    /// How many subscribers took the last published frame, and how many
    /// were too far behind to take it.
    ///
    /// The Python bus dropped silently, so a pipeline losing frames
    /// looked exactly like one producing none. Published as a pair rather
    /// than logged, so a test can assert on it.
    #[getter]
    fn last_publish(&self) -> PyResult<(usize, usize)> {
        let report = self.report.lock().map_err(|_poisoned| unusable("bus"))?;
        Ok((report.delivered, report.dropped))
    }
}

// ---------------------------------------------------------------------
// The two mixins
// ---------------------------------------------------------------------

impl PyInMemoryBus {
    /// Builds the bus itself, for a subclass to stack onto.
    fn over(maxsize: usize) -> Self {
        Self {
            registry: Mutex::new(Vec::new()),
            report: Mutex::new(PublishReport::default()),
            maxsize,
        }
    }
}

/// Mixin that adds bus-publish capability to a pipeline node.
///
/// With no bus attached, `publish` is a silent no-op, so a node can be
/// unit-tested without a live bus.
#[pyclass(subclass, name = "BusPublisher", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyBusPublisher {
    bus: Mutex<Option<Py<PyAny>>>,
}

impl PyBusPublisher {
    /// Builds a publisher with no bus attached.
    ///
    /// Separate from the Python constructor because a subclass built from
    /// Rust passes no arguments, while one built from Python passes its
    /// own and expects them ignored.
    pub(crate) const fn detached() -> Self {
        Self {
            bus: Mutex::new(None),
        }
    }
}

#[pymethods]
impl PyBusPublisher {
    /// Absorbs a subclass calling `super().__init__(...)`.
    ///
    /// A compiled class does its construction in `__new__`, so `__init__`
    /// falls through to `object.__init__`, which refuses arguments. A
    /// Python subclass forwarding its own arguments upward then fails on
    /// a line that worked against the pure-Python base.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Builds a publisher with no bus attached.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    #[pyo3(text_signature = "()")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self::detached()
    }

    /// Attaches the bus published frames are routed through.
    #[pyo3(text_signature = "(bus)")]
    fn attach_bus(&self, bus: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut attached = self
            .bus
            .lock()
            .map_err(|_poisoned| unusable("attached bus"))?;
        *attached = Some(bus.clone().unbind());
        Ok(())
    }

    /// Publishes `frame` to the attached bus, if there is one.
    #[pyo3(text_signature = "(frame)")]
    fn publish(&self, py: Python<'_>, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        let attached = {
            let held = self
                .bus
                .lock()
                .map_err(|_poisoned| unusable("attached bus"))?;
            held.as_ref().map(|bus| bus.clone_ref(py))
        };
        if let Some(bus) = attached {
            bus.bind(py).call_method1("publish", (frame,))?;
        }
        Ok(())
    }
}

/// Mixin that adds bus-subscription capability to a frontend node.
///
/// One queue per subscribed type, kept so that `next_frame` can find it
/// again by type rather than by handle.
#[pyclass(subclass, name = "BusSubscriber", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyBusSubscriber {
    subscriptions: Mutex<Vec<(Py<PyAny>, Py<PyAny>)>>,
}

#[pymethods]
impl PyBusSubscriber {
    /// Absorbs a subclass calling `super().__init__(...)`.
    ///
    /// A compiled class does its construction in `__new__`, so `__init__`
    /// falls through to `object.__init__`, which refuses arguments. A
    /// Python subclass forwarding its own arguments upward then fails on
    /// a line that worked against the pure-Python base.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Builds a subscriber with an empty subscription registry.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    #[pyo3(text_signature = "()")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            subscriptions: Mutex::new(Vec::new()),
        }
    }

    /// Registers a new subscriber queue for `frame_type` on `bus`.
    #[pyo3(text_signature = "(bus, frame_type)")]
    fn subscribe(
        &self,
        py: Python<'_>,
        bus: &Bound<'_, PyAny>,
        frame_type: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let queue = bus.call_method1("subscribe", (frame_type,))?;
        let mut held = self
            .subscriptions
            .lock()
            .map_err(|_poisoned| unusable("subscription registry"))?;
        held.retain(|(known, _queue)| !known.bind(py).is(frame_type));
        held.push((frame_type.clone().unbind(), queue.clone().unbind()));
        Ok(queue.unbind())
    }

    /// Takes the next frame of `frame_type`, or `None`.
    ///
    /// Never raises `queue.Empty`: a caller polling in a render loop gets
    /// `None` instead and needs no exception handling.
    #[pyo3(signature = (frame_type, block = false, timeout = None))]
    #[pyo3(text_signature = "(frame_type, block=False, timeout=None)")]
    fn next_frame(
        &self,
        py: Python<'_>,
        frame_type: &Bound<'_, PyAny>,
        block: bool,
        timeout: Option<f64>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let Some(queue) = self.queue_for(py, frame_type)? else {
            return Ok(None);
        };
        let arguments = PyDict::new(py);
        arguments.set_item("block", block)?;
        arguments.set_item("timeout", timeout)?;
        let empty = py.import("queue")?.getattr("Empty")?;
        match queue.bind(py).call_method("get", (), Some(&arguments)) {
            Ok(frame) => Ok(Some(frame.unbind())),
            Err(raised) if raised.is_instance(py, &empty) => Ok(None),
            Err(raised) => Err(raised),
        }
    }

    /// Drains the queue and returns only the most recent frame.
    ///
    /// What a renderer wants when it falls behind: the current state
    /// rather than the backlog.
    #[pyo3(text_signature = "(frame_type)")]
    fn drain_latest(
        &self,
        py: Python<'_>,
        frame_type: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let Some(queue) = self.queue_for(py, frame_type)? else {
            return Ok(None);
        };
        let empty = py.import("queue")?.getattr("Empty")?;
        let mut latest: Option<Py<PyAny>> = None;
        loop {
            match queue.bind(py).call_method0("get_nowait") {
                Ok(frame) => latest = Some(frame.unbind()),
                Err(raised) if raised.is_instance(py, &empty) => break,
                Err(raised) => return Err(raised),
            }
        }
        Ok(latest)
    }
}

impl PyBusSubscriber {
    /// The queue registered for `frame_type`, if this subscriber has one.
    ///
    /// # Errors
    ///
    /// Returns a `RuntimeError` when the registry cannot be locked.
    fn queue_for(
        &self,
        py: Python<'_>,
        frame_type: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let held = self
            .subscriptions
            .lock()
            .map_err(|_poisoned| unusable("subscription registry"))?;
        Ok(held
            .iter()
            .find(|(known, _queue)| known.bind(py).is(frame_type))
            .map(|(_known, queue)| queue.clone_ref(py)))
    }
}

// ---------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------

/// The work a started node does, which is a Python method call.
///
/// Held by [`arco_runtime::node::Handle`], which owns the thread and
/// catches whatever comes back out of it.
#[derive(Debug)]
struct PythonNode {
    object: Py<PyAny>,
    name: String,
    stop: Arc<AtomicBool>,
}

impl Node for PythonNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&mut self, control: &Control) -> Result<(), Error> {
        // The flag a Python node polls is the one behind the
        // `stop_requested` property, and `PipelineNode.stop` sets that
        // one before it asks the handle to stop, so a node already sees
        // the request by the time the runtime records it. Reading the
        // runtime's flag here keeps the two agreeing for a node that is
        // asked to stop before it has started.
        if control.stop_requested() {
            self.stop.store(true, Ordering::Relaxed);
        }
        Python::attach(|py| {
            self.object
                .call_method0(py, "run")
                .map(|_returned| ())
                .map_err(|raised| {
                    let name = &self.name;
                    let text = raised.to_string();
                    Error::ConflictingArguments {
                        message: format!("node {name} raised: {text}"),
                    }
                })
        })
    }
}

/// The name out of a constructor call, positional or keyword.
///
/// Nothing when the caller passed neither, which is what a subclass with
/// its own argument list does on the way through `__new__`.
fn read_name(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Option<String> {
    if let Some(named) = kwargs.and_then(|given| given.get_item("name").ok().flatten())
        && let Ok(text) = named.extract::<String>()
    {
        return Some(text);
    }
    args.get_item(0)
        .ok()
        .and_then(|first| first.extract::<String>().ok())
}

/// Abstract base class for a single stage in the async pipeline.
///
/// A subclass implements `run`, which is called once on a background
/// thread. A long-running node loops and checks `stop_requested` so that
/// `stop` can return.
#[pyclass(extends = PyBusPublisher, subclass, name = "PipelineNode", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyPipelineNode {
    label: String,
    stop: Arc<AtomicBool>,
    handle: Mutex<Option<Handle>>,
}

#[pymethods]
impl PyPipelineNode {
    /// Takes the name a subclass forwarded upward.
    ///
    /// A compiled class does its construction in `__new__`, which a
    /// Python subclass cannot reach: `__new__` is handed the subclass's
    /// own arguments, and a node written as `MyNode(count=3)` carries no
    /// name at all by then. The name therefore arrives here, on the same
    /// `super().__init__(name=...)` line that set it on the pure-Python
    /// base, and `__new__` accepts whatever the subclass was called with.
    #[pyo3(signature = (*args, **kwargs))]
    fn __init__(&mut self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) {
        if let Some(name) = read_name(args, kwargs) {
            self.label = name;
        }
    }

    /// Builds a node under the given name.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    #[pyo3(text_signature = "(name)")]
    fn new(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyBusPublisher::detached()).add_subclass(Self {
            label: read_name(args, kwargs).unwrap_or_default(),
            stop: Arc::new(AtomicBool::new(false)),
            handle: Mutex::new(None),
        })
    }

    /// The name this node was built under.
    #[getter]
    fn name(&self) -> &str {
        &self.label
    }

    /// Whether the node has been asked to stop.
    #[getter]
    fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Whether the node's thread is still running.
    #[getter]
    fn is_running(&self) -> PyResult<bool> {
        let held = self.handle.lock().map_err(|_poisoned| unusable("node"))?;
        Ok(held.as_ref().is_some_and(Handle::is_running))
    }

    /// How the last run ended, written as a word.
    ///
    /// The Python node swallowed every exception its `run` raised, so a
    /// node that crashed on its first line looked exactly like one that
    /// ran to completion. Published here rather than logged, and beside
    /// the old behavior rather than in place of it: `start` still does
    /// not raise.
    #[getter]
    fn outcome(&self) -> PyResult<String> {
        let held = self.handle.lock().map_err(|_poisoned| unusable("node"))?;
        Ok(match held.as_ref().map(Handle::outcome) {
            None | Some(Outcome::Running) => "running".to_owned(),
            Some(Outcome::Completed) => "completed".to_owned(),
            Some(Outcome::Failed(reason)) => format!("failed: {reason}"),
            Some(Outcome::Panicked(reason)) => format!("panicked: {reason}"),
        })
    }

    /// Starts the node's background thread.
    ///
    /// Calling this on an already-running node does nothing.
    #[pyo3(text_signature = "()")]
    fn start(slf: &Bound<'_, Self>) -> PyResult<()> {
        let node = slf.borrow();
        if node.is_running()? {
            return Ok(());
        }
        node.stop.store(false, Ordering::Relaxed);
        let started = Handle::start(Box::new(PythonNode {
            object: slf.clone().into_any().unbind(),
            name: node.label.clone(),
            stop: Arc::clone(&node.stop),
        }))
        .or_raise()?;
        let mut held = node.handle.lock().map_err(|_poisoned| unusable("node"))?;
        *held = Some(started);
        Ok(())
    }

    /// Asks the node to stop and waits for its thread to exit.
    ///
    /// After this returns the node may be started again.
    #[pyo3(signature = (timeout = None))]
    #[pyo3(text_signature = "(timeout=None)")]
    fn stop(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        self.stop.store(true, Ordering::Relaxed);
        let mut held = self.handle.lock().map_err(|_poisoned| unusable("node"))?;
        let Some(handle) = held.as_mut() else {
            return Ok(());
        };
        let limit = timeout.and_then(|seconds| {
            (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds))
        });
        // The interpreter has to be released here or the node's own
        // thread cannot reach Python to notice the request and return,
        // and the join below would wait forever.
        drop(py.detach(|| handle.stop(limit)));
        Ok(())
    }

    /// Executes the node's work on the background thread.
    #[pyo3(text_signature = "()")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn run(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "PipelineNode.run is abstract.",
        ))
    }
}

/// Orchestrates the async pipeline from a YAML configuration file.
///
/// The display half of the Python runner stays in Python, per deviation
/// A-06, so nothing here imports a window toolkit.
#[pyclass(subclass, name = "PipelineRunner", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyPipelineRunner {
    config: Py<PyDict>,
    bus: Py<PyInMemoryBus>,
    nodes: Mutex<Vec<Py<PyAny>>>,
}

#[pymethods]
impl PyPipelineRunner {
    /// Builds a runner by loading `config_path`.
    #[new]
    #[pyo3(signature = (config_path, bus_maxsize = DEFAULT_CAPACITY))]
    #[pyo3(text_signature = "(config_path, bus_maxsize=64)")]
    fn new(py: Python<'_>, config_path: &Bound<'_, PyAny>, bus_maxsize: usize) -> PyResult<Self> {
        let path = py
            .import("pathlib")?
            .getattr("Path")?
            .call1((config_path,))?;
        if !path.call_method0("exists")?.extract::<bool>()? {
            let shown = path.str()?;
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Pipeline config not found: {shown}"
            )));
        }
        Ok(Self {
            config: load_config(py, &path)?,
            bus: Py::new(py, PyInMemoryBus::new(bus_maxsize))?,
            nodes: Mutex::new(Vec::new()),
        })
    }

    /// The parsed configuration.
    #[getter]
    fn config(&self, py: Python<'_>) -> Py<PyDict> {
        self.config.clone_ref(py)
    }

    /// The shared bus this runner owns.
    #[getter]
    fn bus(&self, py: Python<'_>) -> Py<PyInMemoryBus> {
        self.bus.clone_ref(py)
    }

    /// Registers a node and wires it to the shared bus.
    #[pyo3(text_signature = "(node)")]
    fn register_node(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
        node.call_method1("attach_bus", (self.bus.bind(py),))?;
        let mut held = self.nodes.lock().map_err(|_poisoned| unusable("runner"))?;
        held.push(node.clone().unbind());
        Ok(())
    }

    /// Registers a frontend subscriber for `frame_type`.
    ///
    /// Safe before or after `start`, which is the late-subscriber support
    /// the Python runner advertised.
    #[pyo3(text_signature = "(subscriber, frame_type)")]
    fn attach_subscriber(
        &self,
        py: Python<'_>,
        subscriber: &Bound<'_, PyAny>,
        frame_type: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        subscriber.call_method1("subscribe", (self.bus.bind(py), frame_type))?;
        Ok(())
    }

    /// Starts every registered node that is not already running.
    #[pyo3(text_signature = "()")]
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        for node in self.registered(py)? {
            if !node.bind(py).getattr("is_running")?.extract::<bool>()? {
                node.bind(py).call_method0("start")?;
            }
        }
        Ok(())
    }

    /// Stops every registered node, in reverse registration order.
    #[pyo3(signature = (timeout = None))]
    #[pyo3(text_signature = "(timeout=None)")]
    fn stop(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        let arguments = PyDict::new(py);
        arguments.set_item("timeout", timeout)?;
        for node in self.registered(py)?.into_iter().rev() {
            node.bind(py).call_method("stop", (), Some(&arguments))?;
        }
        Ok(())
    }

    /// The nodes registered so far, in registration order.
    #[getter]
    fn nodes(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, self.registered(py)?)?.unbind())
    }
}

impl PyPipelineRunner {
    /// A copy of the node list, so the lock is not held across a call
    /// back into Python.
    ///
    /// # Errors
    ///
    /// Returns a `RuntimeError` when the node list cannot be locked.
    fn registered(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let held = self.nodes.lock().map_err(|_poisoned| unusable("runner"))?;
        Ok(held.iter().map(|node| node.clone_ref(py)).collect())
    }
}

/// Parses a configuration file as YAML.
///
/// Anything that is not a mapping, an empty file included, reads as an
/// empty configuration, which is what the Python runner did.
///
/// # Errors
///
/// Returns a `RuntimeError` when `pyyaml` is absent, and otherwise
/// whatever reading or parsing the file raised.
fn load_config(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let Ok(yaml) = py.import("yaml") else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "pyyaml is required to load pipeline configuration files. \
             Install it with: pip install pyyaml",
        ));
    };
    let arguments = PyDict::new(py);
    arguments.set_item("encoding", "utf-8")?;
    let text = path.call_method("read_text", (), Some(&arguments))?;
    let parsed = yaml.call_method1("safe_load", (text,))?;
    Ok(parsed
        .cast_into::<PyDict>()
        .unwrap_or_else(|_not_a_mapping| PyDict::new(py))
        .unbind())
}

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------

/// Adds every middleware and pipeline name to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised, which the interpreter
/// surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBus>()?;
    module.add_class::<PyInMemoryBus>()?;
    module.add_class::<PyBusPublisher>()?;
    module.add_class::<PyBusSubscriber>()?;
    module.add_class::<PyPipelineNode>()?;
    module.add_class::<PyPipelineRunner>()?;
    Ok(())
}
