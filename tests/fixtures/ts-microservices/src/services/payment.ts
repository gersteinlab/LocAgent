import { BaseService, db, publish, EventType } from '../shared';
import type { PaymentStatus } from '../types';

export class PaymentService extends BaseService {
  constructor() {
    super('payment');
  }

  authorize(amount: number): PaymentStatus {
    this.log('authorize');
    db.query('insert into payments', [amount]);
    publish({ type: EventType.PaymentAuthorized, payload: { amount } });
    return 'authorized';
  }
}
