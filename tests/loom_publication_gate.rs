//! Exhaustive scheduler proofs for the shared latest-only worker contract.

#[path = "../src/publication_gate.rs"]
mod publication_gate;

use loom::sync::{Arc, Mutex};
use loom::thread;
use publication_gate::LatestOnlyPublicationGate;

#[test]
fn stale_completion_never_overwrites_the_newest_publication() {
    loom::model(|| {
        let gate = Arc::new(Mutex::new(LatestOnlyPublicationGate::default()));
        let first = gate.lock().unwrap().request();
        let first_worker = Arc::clone(&gate);
        let stale = thread::spawn(move || first_worker.lock().unwrap().try_publish(first));

        let second = gate.lock().unwrap().request();
        let second_worker = Arc::clone(&gate);
        let newest = thread::spawn(move || second_worker.lock().unwrap().try_publish(second));

        let _ = stale.join().unwrap();
        let _ = newest.join().unwrap();
        assert_eq!(
            gate.lock().unwrap().published_generation(),
            Some(second.generation())
        );
    });
}

#[test]
fn cancellation_invalidates_every_already_claimed_token() {
    loom::model(|| {
        let gate = Arc::new(Mutex::new(LatestOnlyPublicationGate::default()));
        let token = gate.lock().unwrap().request();
        let worker_gate = Arc::clone(&gate);
        let worker = thread::spawn(move || {
            thread::yield_now();
            worker_gate.lock().unwrap().try_publish(token)
        });

        gate.lock().unwrap().cancel_all();
        let published = worker.join().unwrap();
        if published {
            // The worker won before cancellation. The barrier still clears the
            // visible request and no callback can republish the old token.
            assert!(!gate.lock().unwrap().try_publish(token));
        } else {
            assert_eq!(gate.lock().unwrap().published_generation(), None);
        }
    });
}

#[test]
fn claim_coalesces_a_burst_to_the_latest_generation() {
    loom::model(|| {
        let gate = Arc::new(Mutex::new(LatestOnlyPublicationGate::default()));
        let first = gate.lock().unwrap().request();
        let second = gate.lock().unwrap().request();
        let claimed = gate.lock().unwrap().claim_latest().unwrap();
        assert_eq!(claimed, second);
        assert!(!gate.lock().unwrap().try_publish(first));
        assert!(gate.lock().unwrap().try_publish(claimed));
        assert_eq!(
            gate.lock().unwrap().published_generation(),
            Some(second.generation())
        );
    });
}

#[test]
fn one_token_has_exactly_one_publication_owner() {
    loom::model(|| {
        let gate = Arc::new(Mutex::new(LatestOnlyPublicationGate::default()));
        let token = gate.lock().unwrap().request();
        let first_gate = Arc::clone(&gate);
        let second_gate = Arc::clone(&gate);
        let first = thread::spawn(move || first_gate.lock().unwrap().try_publish(token));
        let second = thread::spawn(move || second_gate.lock().unwrap().try_publish(token));

        let first_owned = first.join().unwrap();
        let second_owned = second.join().unwrap();
        assert_ne!(
            first_owned, second_owned,
            "exactly one completion must own publication"
        );
        assert_eq!(
            gate.lock().unwrap().published_generation(),
            Some(token.generation())
        );
    });
}

#[test]
#[should_panic(expected = "broken gate permits duplicate publication ownership")]
fn deliberately_broken_publication_ownership_is_caught() {
    loom::model(|| {
        // This is the smallest mutation of the ownership law: a completion
        // checks that its token is current but forgets to consume the token.
        // Keeping the mutant local makes the production gate impossible to
        // select while proving the scheduler harness rejects the bug.
        let current = Arc::new(Mutex::new(Some(1_u64)));
        let first_current = Arc::clone(&current);
        let second_current = Arc::clone(&current);
        let first = thread::spawn(move || *first_current.lock().unwrap() == Some(1));
        let second = thread::spawn(move || *second_current.lock().unwrap() == Some(1));

        let owners = usize::from(first.join().unwrap()) + usize::from(second.join().unwrap());
        assert_eq!(
            owners, 1,
            "broken gate permits duplicate publication ownership"
        );
    });
}
