export class BaseService {
  protected serviceName: string;

  constructor(serviceName: string) {
    this.serviceName = serviceName;
  }

  protected log(message: string): void {
    console.log(`[${this.serviceName}] ${message}`);
  }
}
